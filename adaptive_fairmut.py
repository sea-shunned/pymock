import precompute
import initialisation
import objectives
import operators
import classes
import evaluation
import numpy as np
from itertools import count

from os import cpu_count

from deap import base
from deap import creator
from deap import tools
from deap.benchmarks.tools import hypervolume

import multiprocessing

# Run outside of multiprocessing scope
creator.create("Fitness", base.Fitness, weights=(-1.0, -1.0)) #(VAR, CNN)
creator.create("Individual", list, fitness=creator.Fitness)

# @profile # for line_profiler
def main(data_path, delta_int_links, HV_ref):
	print("Delta:",delta_int_links)

	# If labels are present (will move to a argparse thing eventually)
	labels = True

	# Load data
	data, data_dict = classes.createDataset(data_path, labels)

	######## Parameters ########
	# Neighbourhood size
	L = 10

	# Population size
	num_indivs = 100

	# Reduced genotype length
	relev_links_len = initialisation.relevantLinks(delta_int_links, classes.Dataset.num_examples)
	# fixed_links_len = num_examples - relev_links_len

	#### relev_links_len needs a rename to more accurately describe that it is the reduced genotype length

	# k_user is the double the expected number of clusters in the dataset
	k_user = 20 # will move to a argparse thing eventually

	############## TO DO ##############
	######## Precomputation ########
	### Should I consider putting all of the below in it's own script/function?
	### A precomputation function or something that does all of this
	### Or at least abstract out this vars_check business into precomputation

	import time # Just to see how long sections take

	distarray = precompute.compDists(data,data)
	distarray = precompute.normaliseDistArray(distarray)
	argsortdists = np.argsort(distarray, kind='mergesort')
	nn_rankings = precompute.nnRankings(distarray, classes.Dataset.num_examples)
	mst_genotype = precompute.createMST(distarray)

	degree_int = precompute.degreeInterest(mst_genotype, L, nn_rankings, distarray)

	# Return an array with the indices of the most interesting links first
	int_links_indices = precompute.interestLinksIndices(degree_int)

	base_genotype, base_clusters = initialisation.baseGenotype(mst_genotype, int_links_indices, relev_links_len)
	part_clust, cnn_pairs = classes.partialClustering(base_clusters, data, data_dict, argsortdists, L)
	conn_array, max_conn = classes.PartialClust.conn_array, classes.PartialClust.max_conn

	# print(conn_array)
	# print(max_conn)

	# Consider removing the [:relev_links_len] slice incase we change the delta value
	reduced_clust_nums = [data_dict[i].base_cluster_num for i in int_links_indices[:relev_links_len]]


	######## Population Initialisation ########
	toolbox = base.Toolbox()

	toolbox.register("initDelta", initialisation.initDeltaMOCK, k_user, num_indivs, mst_genotype, int_links_indices, relev_links_len, argsortdists, L)
	toolbox.register("population", tools.initIterate, list, toolbox.initDelta)

	toolbox.register("evaluate", objectives.evalMOCK, part_clust = part_clust, reduced_clust_nums = reduced_clust_nums, conn_array = conn_array, max_conn = max_conn, num_examples = classes.Dataset.num_examples, data_dict=data_dict, cnn_pairs=cnn_pairs)
	# In the new paper they put the crossover probability as 1
	toolbox.register("mate", operators.uniformCrossover, cxpb = 1.0)
	# We just use the MUTPB = 1 in the (1/num-examples) term, as per the Garza/Handl code
	toolbox.register("mutate", operators.neighbourMutation, MUTPB = 1.0, gen_length = relev_links_len, argsortdists=argsortdists, L = L, int_links_indices=int_links_indices, nn_rankings = nn_rankings)
	# DEAP has a built-in selection tool for NSGA2
	toolbox.register("select", tools.selNSGA2)
	# For multiprocessing
	pool = multiprocessing.Pool(processes = cpu_count()-2)
	toolbox.register("map", pool.map)
	toolbox.register("starmap", pool.starmap)

	# They do use a stats module which I'll need to look at
	# Perhaps integrate the gap statistic/rand index evaluation stuff into it?

	NUM_GEN = 100 # 100 in Garza/Handl
	# CXPB = 1.0 # 1.0 in Garza/Handl i.e. always crossover
	MUTPB = 1.0 # 1.0 in Garza/Handl i.e. always enter mutation, indiv link prob is calculated there
	NUM_INDIVS = 100 # 100 in Garza/Handl
	
	init_pop_start = time.time()
	pop = toolbox.population()
	init_pop_end = time.time()
	print("Initial population:",init_pop_end - init_pop_start)

	# Convert each individual of class list to class deap.creator.Individual
	# Easier than modifying population function(s)
	for index, indiv in enumerate(pop):
		indiv = creator.Individual(indiv)
		pop[index] = indiv

	# That is how https://github.com/DEAP/deap/blob/master/examples/ga/nsga2.py does it
	# Should check more examples to see if this is THE way or just a way

	# Need to evaluate the initial population
	VAR_init = [] # For graphical comparison
	CNN_init = []
	fitnesses = toolbox.map(toolbox.evaluate, pop)
	for ind, fit in zip(pop, fitnesses):
		ind.fitness.values = fit
		VAR_init.append(fit[0])
		CNN_init.append(fit[1])

	if HV_ref == None:
		# max_conn varies a lot with delta, so *2 here just in case
		# If we engage adaptive, will want a check to see if new max_conn is about of bounds
			# then provide a warning
		HV_ref = [np.ceil(np.max(VAR_init)*2), np.ceil(max_conn*2)]
	print("HV Ref:",HV_ref)

	if max_conn >= HV_ref[1]:
		print(max_conn, HV_ref[1])
		raise ValueError("Max CNN value has exceeded that set for HV reference point, HV values may be unreliable")
		# print("Max CNN value has exceeded that set for HV reference point, HV values may be unreliable")
		print("Continuing...\n")

	# This is just to assign the crowding distance to the individuals, no actual selection is done
	# Accessed by individual.fitness.crowding_dist
	pop = toolbox.select(pop, len(pop))

	# Create stats object
	# Set the key to be the fitness values of the individual
	stats = tools.Statistics(lambda ind: ind.fitness.values)
	stats.register("avg", np.mean, axis=0)
	stats.register("std", np.std, axis=0)
	stats.register("min", np.min, axis=0)
	stats.register("max", np.max, axis=0)

	logbook = tools.Logbook()
	logbook.header = "gen", "evals", "std", "min", "avg", "max"

	record = stats.compile(pop)
	logbook.record(gen=0, evals=len(pop), **record)

	### Adaptive hyperparameter parameters ###
	window_size = 3 			# Moving average of gradients to look at
	initial_gens = 10 			# Number of generations to wait until measuring
	init_grad_switch = True 	# To calculate initial gradient only once
	new_delta_window = 5		# Number of generations to let new delta take effect before triggering new
	adapt_gens = [0]			# Initialise list for tracking which gens we trigger adaptive delta
	HV = []						# Initialise HV list
	grads = [0]					# Initialise gradient list

	# Calculate HV of initialised population
	HV.append(hypervolume(pop, HV_ref))

	ea_start = time.time()
	for gen in range(1, NUM_GEN):
		offspring = tools.selTournamentDCD(pop, len(pop))
		# offspring = [toolbox.clone(ind) for ind in offspring]
		offspring = toolbox.map(toolbox.clone,offspring) # Map version of above, should be same

		# If done properly, using comprehensions/map should speed this up
		for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
			# We actually check if crossover should occur in the mate function
			# CXPB will pretty much always be one so no need for an if, just send straight to crossover

			# test = [ind1[i] for i in int_links_indices[:relev_links_len]]
			# print(type(ind1), type([ind1[i] for i in int_links_indices[:relev_links_len]]), type(test))

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
		fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
		for ind, fit in zip(invalid_ind, fitnesses):
			ind.fitness.values = fit

		pop = toolbox.select(pop + offspring, NUM_INDIVS)

		print("Gen:",gen)
		curr_HV = hypervolume(pop, HV_ref)
		HV.append(curr_HV)

		if delta_int_links != 0:
			if gen > initial_gens:
				if init_grad_switch:
					init_grad = (HV[-1] - HV[0]) / len(HV)
					init_grad_switch = False
					# print(init_grad)

				grads.append((curr_HV - HV[-(window_size+1)]) / window_size)

				# Just in case, capture the error (unlikely to happen with large datasets)
				try:
					curr_ratio = grads[-2]/grads[-1]

				except ZeroDivisionError:
					print("Gradient zero division error, using 0.0001")
					curr_ratio = grads[-2]/0.0001

				# print(curr_ratio)

				if gen >= adapt_gens[-1] + new_delta_window:
					if ((np.around(curr_ratio, decimals=2) == 1
						or np.around(curr_ratio, decimals=2) == 0) 
						# or grads[-1] < 0.01 * init_grad
						and grads[-1] < 0.5 * init_grad):
						adapt_gens.append(gen)

						##### New computation goes here #####
						toolbox.unregister("evaluate")
						toolbox.unregister("mutate")
						# Reset the partial clust counter to ceate new base clusters
						classes.PartialClust.id_value = count()

						# Reduce delta value
						relev_links_len_old = relev_links_len
						delta_int_links -= 5


						print("Adaptive Delta engaged! Going down to delta =", delta_int_links)

						# print("Before:",len(part_clust))

						# Need to look at resetting part_clust, as it is just growing!

						# Re-do the relevant precomputation
						relev_links_len = initialisation.relevantLinks(delta_int_links, classes.Dataset.num_examples)
						base_genotype, base_clusters = initialisation.baseGenotype(mst_genotype, int_links_indices, relev_links_len)
						part_clust, cnn_pairs = classes.partialClustering(base_clusters, data, data_dict, argsortdists, L)
						conn_array, max_conn = classes.PartialClust.conn_array, classes.PartialClust.max_conn
						reduced_clust_nums = [data_dict[i].base_cluster_num for i in int_links_indices[:relev_links_len]]
					
						# print("After:",len(part_clust))

						# Re-register the relevant functions with changed arguments
						toolbox.register("evaluate", objectives.evalMOCK, part_clust = part_clust, reduced_clust_nums = reduced_clust_nums, conn_array = conn_array, max_conn = max_conn, num_examples = classes.Dataset.num_examples, data_dict=data_dict, cnn_pairs=cnn_pairs)
						# toolbox.register("mutate", operators.neighbourMutation, MUTPB = 1.0, gen_length = relev_links_len, argsortdists=argsortdists, L = L, int_links_indices=int_links_indices, nn_rankings = nn_rankings)

						# Use new fair mutation strategy
						toolbox.register("mutate", operators.neighbourMutationAdapt, MUTPB = 1.0, gen_length = relev_links_len, argsortdists=argsortdists, L = L, int_links_indices=int_links_indices, nn_rankings = nn_rankings, old_length = relev_links_len_old)

						# Maybe check if it has been x generations since last adaptive change
							# So we can switch back to normal mutation scheme

						newly_unfixed_indices = int_links_indices[relev_links_len_old:relev_links_len]
						for indiv in pop:
							indiv.extend([mst_genotype[i] for i in newly_unfixed_indices])

		record = stats.compile(pop)
		logbook.record(gen=gen, evals=len(invalid_ind), **record)

	print("Adaptive engaged at gens:",adapt_gens[1:])
	ea_end = time.time()
	ea_time = ea_end - ea_start
	print("EA time:", ea_time)
	print("Final population hypervolume is %f" % hypervolume(pop, HV_ref))

	# print(len(grads))

	## Unregister alias funcs - needed?
	# toolbox.unregister("pop_init")
	# toolbox.unregister("population")
	toolbox.unregister("evaluate")
	# toolbox.unregister("mate")
	# toolbox.unregister("mutate")
	# toolbox.unregister("select")

	# Close pools just in case (shouldn't be needed)
	pool.close()
	pool.join()

	final_pop_metrics = evaluation.finalPopMetrics(pop, mst_genotype, int_links_indices, relev_links_len)

	classes.PartialClust.id_value = count()
	
	return pop, logbook, VAR_init, CNN_init, HV, ea_time, HV_ref, final_pop_metrics

if __name__ == "__main__":
	import os
	import pandas as pd

	base_path = os.getcwd()
	data_folder = base_path+"/data/data_handl/"
	data_name = "10d-20c-no0.dat"
	# data_folder = base_path+"/data/iris/"
	# data_name = "iris.csv"
	data_path = data_folder+data_name




	#### Adaptive Delta Testing ####
	# To test this properly, I will need to save an initial population
	# Then use multiple runs from this to see if the delta helps
	# I'll need to use a few different initial populations of course
	# Need to do this after I have fixed/checked/compared the initialisation schemes

	# Will probably need to unregister and register the evaluation function etc. again
	# As some of the arguments that have been frozen no longer hold




	########### TO DO ###########
	# Identify what we would need to re-run in a adaptive delta setting
	# Look at notes in classes.py to implement the hyperparameter stuff

