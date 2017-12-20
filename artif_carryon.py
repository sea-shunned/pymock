import precompute
import initialisation
import objectives
import operators
import classes
import evaluation
import numpy as np
from itertools import count
import random
from adaptive_funcs import triggerGens

# For multiprocessing
from os import cpu_count
import multiprocessing

from deap import base
from deap import creator
from deap import tools
from deap.benchmarks.tools import hypervolume

# To measure sections
import time

# Run outside of multiprocessing scope
# creator.create("Fitness", base.Fitness, weights=(-1.0, -1.0)) #(VAR, CNN)
# creator.create("Individual", list, fitness=creator.Fitness)
## Only need to do the above once, which we do with main_base.py

# @profile # for line_profiler
def main(data, data_dict, delta_val, HV_ref, argsortdists, nn_rankings, mst_genotype, int_links_indices, L, num_indivs, num_gens, delta_reduce):

	# Reduced genotype length
	relev_links_len = initialisation.relevantLinks(delta_val, classes.Dataset.num_examples)
	print("Genotype length:",relev_links_len)

	#### relev_links_len needs a rename to more accurately describe that it is the reduced genotype length

	# Could abstract some of this out so that for big experiments it is only run once per dataset

	base_genotype, base_clusters = initialisation.baseGenotype(mst_genotype, int_links_indices, relev_links_len)
	part_clust, cnn_pairs = classes.partialClustering(base_clusters, data, data_dict, argsortdists, L)
	conn_array, max_conn = classes.PartialClust.conn_array, classes.PartialClust.max_conn
	reduced_clust_nums = [data_dict[i].base_cluster_num for i in int_links_indices[:relev_links_len]]

	######## Population Initialisation ########
	toolbox = base.Toolbox()

	toolbox.register("initDelta", initialisation.initDeltaMOCK, classes.Dataset.k_user, num_indivs, mst_genotype, int_links_indices, relev_links_len, argsortdists, L)
	toolbox.register("population", tools.initIterate, list, toolbox.initDelta)

	toolbox.register("evaluate", objectives.evalMOCK, part_clust = part_clust, reduced_clust_nums = reduced_clust_nums, conn_array = conn_array, max_conn = max_conn, num_examples = classes.Dataset.num_examples, data_dict=data_dict, cnn_pairs=cnn_pairs, base_members=classes.PartialClust.base_members, base_centres=classes.PartialClust.base_centres)
	# In the new paper they put the crossover probability as 1
	toolbox.register("mate", operators.uniformCrossover, cxpb = 1.0)
	# We just use the MUTPB = 1 in the (1/num-examples) term, as per the Garza/Handl code
	toolbox.register("mutate", operators.neighbourMutation, MUTPB = 1.0, gen_length = relev_links_len, argsortdists=argsortdists, L = L, int_links_indices=int_links_indices, nn_rankings = nn_rankings)
	# DEAP has a built-in selection tool for NSGA2
	toolbox.register("select", tools.selNSGA2)
	# For multiprocessing
	pool = multiprocessing.Pool(processes = cpu_count())
	toolbox.register("map", pool.map, chunksize=20)
	# toolbox.register("starmap", pool.starmap)

	# They do use a stats module which I'll need to look at
	# Perhaps integrate the gap statistic/rand index evaluation stuff into it?

	######## Parameters ########
	# NUM_GEN = 100 # 100 in Garza/Handl
	# CXPB = 1.0 # 1.0 in Garza/Handl i.e. always crossover
	# MUTPB = 1.0 # 1.0 in Garza/Handl i.e. always enter mutation, indiv link prob is calculated there
	
	pop = toolbox.population()

	# Convert each individual of class list to class deap.creator.Individual
	# Easier than modifying population function
	# Remember I will need to do this for the reinit strategy
	for index, indiv in enumerate(pop):
		indiv = creator.Individual(indiv)
		pop[index] = indiv

	# That is how https://github.com/DEAP/deap/blob/master/examples/ga/nsga2.py does it

	# Evaluate the initial population
	VAR_init = []
	CNN_init = []
	fitnesses = toolbox.map(toolbox.evaluate, pop)
	for ind, fit in zip(pop, fitnesses):
		ind.fitness.values = fit	
		VAR_init.append(fit[0])
		CNN_init.append(fit[1])

	if HV_ref == None:
		# max_conn varies a lot with delta, so start with lowest delta
		# 5% breathing room used
		# MST should have highes VAR value possible, regardless of delta
		# Check this so that we don't give too much unnecessary room
		HV_ref = [np.ceil(np.max(VAR_init)*1.5), np.ceil(max_conn+1)]

	# Check that our current HV_ref is always valid
	# If we just start with delta=0 (or the lowest delta value) then this won't be a problem...
	if max_conn >= HV_ref[1]:
		print(max_conn, HV_ref[1])
		# raise ValueError("Max CNN value has exceeded that set for HV reference point, HV values may be unreliable")
		print("Max CNN value has exceeded that set for HV reference point, HV values may be unreliable")
		print("Continuing...\n")

	# This is just to assign the crowding distance to the individuals, no actual selection is done
	# Accessed by individual.fitness.crowding_dist
	pop = toolbox.select(pop, len(pop))

	# Create stats object
	# Set the key to be the fitness values of the individual
	# stats = tools.Statistics(lambda ind: ind.fitness.values)
	# stats.register("avg", np.mean, axis=0)
	# stats.register("std", np.std, axis=0)
	# stats.register("min", np.min, axis=0)
	# stats.register("max", np.max, axis=0)

	# logbook = tools.Logbook()
	# logbook.header = "gen", "evals", "avg", "std", "min", "max"

	# record = stats.compile(pop)
	# logbook.record(gen=0, evals=len(pop), **record)

	# Calculate HV of initialised population
	HV = []
	HV.append(hypervolume(pop, HV_ref))

	### Adaptive hyperparameter parameters ###
	window_size = 3				# Moving average of gradients to look at
	block_trigger_gens = 10		# Number of generations to wait until measuring
	adapt_gens = [0]			# Initialise list for tracking which gens we trigger adaptive delta

	# print(np.sum([ind.fitness.values for ind in pop]))

	trigger_gens = triggerGens(num_gens)

	### Start actual EA ### 
	ea_start = time.time()
	for gen in range(1, num_gens):
		# Shuffle population
		random.shuffle(pop)

		offspring = tools.selTournamentDCD(pop, len(pop))
		# offspring = [toolbox.clone(ind) for ind in offspring]
		offspring = toolbox.map(toolbox.clone,offspring) # Map version of above, should be same

		# If done properly, using comprehensions/map should speed this up
		for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
			# print(id(ind1),id(ind2))
			# CXPB will pretty much always be one so no need for an if, just send straight to crossover
			toolbox.mate(ind1, ind2)

			# Mutate individuals
			toolbox.mutate(ind1)
			toolbox.mutate(ind2)

		### The below takes longer! Odd, surely it should be faster if we do it right # was this due to profiling?
		# mapped_off = pool.starmap(toolbox.mate,zip(offspring[::2], offspring[1::2]))
		# offspring = [gen for i in mapped_off for gen in i]
		# offspring = toolbox.map(toolbox.mutate,offspring)

		# Evaluate fitness of those that are missing a value
		invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

		assert len(invalid_ind) == num_indivs, "Some individuals are unchanged?"

		fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
		for ind, fit in zip(invalid_ind, fitnesses):
			ind.fitness.values = fit

		pop = toolbox.select(pop + offspring, num_indivs)

		# print("Gen:",gen)
		HV.append(hypervolume(pop, HV_ref))

		### Adaptive Delta Trigger ###
		if delta_val != 0:
			if gen in trigger_gens:
				print("Triggered at:",gen)
				# Re-do the relevant precomputation
				toolbox.unregister("evaluate")
				toolbox.unregister("mutate")

				# Reset the partial clust counter to ceate new base clusters
				classes.PartialClust.id_value = count()

				# Save old genotype length
				relev_links_len_old = relev_links_len
				
				# Reduce delta by flat value or multiple of the square root if using that
				if isinstance(delta_val, int):
					delta_val -= delta_reduce
				else:
					delta_val -= (100*delta_reduce*np.sqrt(classes.Dataset.num_examples))/classes.Dataset.num_examples

				print("Adaptive Delta engaged at gen {}! Going down to delta = {}".format(gen, delta_val))

				# Re-do the relevant precomputation
				relev_links_len = initialisation.relevantLinks(delta_val, classes.Dataset.num_examples)
				print("Genotype length:",relev_links_len)
				base_genotype, base_clusters = initialisation.baseGenotype(mst_genotype, int_links_indices, relev_links_len)
				part_clust, cnn_pairs = classes.partialClustering(base_clusters, data, data_dict, argsortdists, L)
				conn_array, max_conn = classes.PartialClust.conn_array, classes.PartialClust.max_conn
				reduced_clust_nums = [data_dict[i].base_cluster_num for i in int_links_indices[:relev_links_len]]
			
				newly_unfixed_indices = int_links_indices[relev_links_len_old:relev_links_len]
				# print(newly_unfixed_indices)
				for indiv in pop:
					indiv.extend([mst_genotype[i] for i in newly_unfixed_indices])
				
				# Re-register the relevant functions with changed arguments
				toolbox.register("evaluate", objectives.evalMOCK, part_clust = part_clust, reduced_clust_nums = reduced_clust_nums, conn_array = conn_array, max_conn = max_conn, num_examples = classes.Dataset.num_examples, data_dict=data_dict, cnn_pairs=cnn_pairs, base_members=classes.PartialClust.base_members, base_centres=classes.PartialClust.base_centres)
				toolbox.register("mutate", operators.neighbourMutation, MUTPB = 1.0, gen_length = relev_links_len, argsortdists=argsortdists, L = L, int_links_indices=int_links_indices, nn_rankings = nn_rankings)


		# record = stats.compile(pop)
		# logbook.record(gen=gen, evals=len(invalid_ind), **record)

	ea_end = time.time()
	ea_time = ea_end - ea_start
	print("EA time:", ea_time)
	print("Final population hypervolume is",HV[-1])

	# print(HV)
	print("Triggered gens:",adapt_gens)

	# print(logbook)
	# print(len(tools.sortNondominated(pop, len(pop))))
	# print(tools.sortNondominated(pop, len(pop))[0])
	# print(len(tools.sortNondominated(pop, len(pop))[0])) # if ==len(pop) then only one front

	# Close pools just in case (shouldn't be needed)
	pool.close()
	pool.join()

	# Reset the cluster ID value if we're running multiple values
	# Alternate solution is to reload the module
	classes.PartialClust.id_value = count()

	return pop, HV, HV_ref, int_links_indices, relev_links_len, adapt_gens