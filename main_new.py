import precompute
import initialisation
import objectives
import operators
import classes

# For multiprocessing
# from os import cpu_count
import multiprocessing

from deap import base, creator, tools
from deap.benchmarks.tools import hypervolume

import random

# Run outside of multiprocessing scope
creator.create("Fitness", base.Fitness, weights=(-1.0, -1.0)) #(VAR, CNN)
creator.create("Individual", list, fitness=creator.Fitness, fairmut=None)

def delta_precomp(delta_val, mst_genotype, int_links_indices):
    relev_links_len = initialisation.relevantLinks(
        delta_val, classes.Dataset.num_examples)
    print("Genotype length:", relev_links_len)

    base_genotype, base_clusters = initialisation.baseGenotype(
        mst_genotype, int_links_indices, relev_links_len)

    classes.partialClustering(
        base_clusters, data, data_dict, argsortdists, L)

    # Maybe also put this as a class attribute for PartialClust?
    reduced_clust_nums = [
    data_dict[i].base_cluster_num for i in int_links_indices[:relev_links_len]
    ]

    return relev_links_len, reduced_clust_nums

# Consider trying to integrate the use of **kwargs here
def create_base_toolbox(num_indivs, mst_genotype, int_links_indices, relev_links_len, argsortdists, L, data_dict, nn_rankings):
    toolbox = base.Toolbox()

    # Register the initialisation function
    toolbox.register(
        "initDelta", 
        initialisation.initDeltaMOCK, 
        k_user = classes.Dataset.k_user, 
        num_indivs = num_indivs, 
        mst_genotype = mst_genotype, 
        int_links_indices = int_links_indices, 
        relev_links_len = relev_links_len, 
        argsortdists = argsortdists, 
        L = L
        )

    # Register the population function that uses the custom initialisation
    toolbox.register(
        "population", 
        tools.initIterate, 
        list, 
        toolbox.initDelta
        )

    # Register the evaluation function
    toolbox.register(
        "evaluate", 
        objectives.evalMOCK, 
        part_clust = classes.PartialClust.part_clust, 
        reduced_clust_nums = reduced_clust_nums, 
        conn_array = classes.PartialClust.conn_array, 
        max_conn = classes.PartialClust.max_conn, 
        num_examples = classes.Dataset.num_examples, 
        data_dict = data_dict, 
        cnn_pairs = classes.PartialClust.cnn_pairs, 
        base_members = classes.PartialClust.base_members, 
        base_centres = classes.PartialClust.base_centres
        )

    # Register the crossover function
    toolbox.register(
        "mate", 
        operators.uniformCrossover, 
        cxpb = 1.0
        )

    # Register the mutation function
    toolbox.register(
        "mutate", operators.neighbourMutation, 
        MUTPB = 1.0, 
        gen_length = relev_links_len, 
        argsortdists = argsortdists, 
        L = L, 
        int_links_indices = int_links_indices, 
        nn_rankings = nn_rankings
        )
    
    # Register the selection function (built-in with DEAP for NSGA2)
    toolbox.register(
        "select", 
        tools.selNSGA2
        )
    
    return toolbox

def initial_setup(toolbox):
    pop = toolbox.population()

	# Convert each individual of class list to class deap.creator.Individual
	# Easier than modifying population function
    for index, indiv in enumerate(pop):
        indiv = creator.Individual(indiv)
        pop[index] = indiv

    # Lists to capture the initial population fitness (if desired)
    VAR_init = []
    CNN_init = []

    # Evaluate the initial pop
    fitnesses = toolbox.map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit	
        VAR_init.append(fit[0])
        CNN_init.append(fit[1])
    
    # This is just to assign the crowding distance to the individuals, no actual selection is done
    pop = toolbox.select(pop, len(pop))

    return pop, VAR_init, CNN_init

def generation(pop, toolbox, HV, HV_ref):
    # Shuffle the population
    random.shuffle(pop)

    # Select and clone the offspring (to allow modification)
    offspring = tools.selTournamentDCD(pop, len(pop))
    offspring = toolbox.map(toolbox.clone, offspring)

    # Tournament
    for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
        # Crossover
        toolbox.mate(ind1, ind2)

        # Mutation
        toolbox.mutate(ind1)
        toolbox.mutate(ind2)

    # Evaluate the offspring
    fitnesses = toolbox.map(toolbox.evaluate, offspring)
    for ind, fit in zip(offspring, fitnesses):
        ind.fitness.values = fit

    # Select from the current population and new offspring
    pop = toolbox.select(pop + offspring, num_indivs)

    HV.append(hypervolume(pop, HV_ref))

    return pop, HV

def reinit_generation(pop, toolbox, HV, HV_ref):
    random.shuffle(pop)

    # Generate the offspring from the initialisation routine
    offspring = toolbox.population()
    for index, indiv in enumerate(offspring):
        indiv = creator.Individual(indiv)
        offspring[index] = indiv

    # Evaluate the offspring
    fitnesses = toolbox.map(toolbox.evaluate, offspring)
    for ind, fit in zip(offspring, fitnesses):
        ind.fitness.values = fit

    # Select from the current population and new offspring
    pop = toolbox.select(pop + offspring, num_indivs)

    HV.append(hypervolume(pop, HV_ref))   

    return pop, HV

def calc_HV_ref(VAR_init):
    return [np.ceil(np.max(VAR_init)*1.5), np.ceil(classes.PartialClust.max_conn+1)]

def check_trigger(delta_val, gen, adapt_gens, max_adapts, block_trigger_gens, HV, window_size, ref_grad):
    if delta_val != 0:
        if len(adapt_gens) < max_adapts:
            if gen >= (adapt_gens[-1] + block_trigger_gens):
                if gen == (adapt_gens[-1] + block_trigger_gens):
                    ref_grad = (HV[-1] - HV[adapt_gens[-1]]) / len(HV)
                    print("Reference gradient:", ref_grad, "at gen", gen)
                    return False, ref_grad

                curr_grad = (HV[-1] - HV[-(window_size+1)]) / window_size

                # Debugging, to remove
                if ref_grad is None:
                    raise ValueError("ref_grad is None")

                if curr_grad <= 0.1 * ref_grad:
                    return True, ref_grad

    return False, ref_grad

def adaptive_delta_trigger(pop, gen, strat_name, delta_val, toolbox, delta_reduce, relev_links_len, mst_genotype, int_links_indices, num_indivs, argsortdists, L, data_dict, nn_rankings):

    # Need to re-register the functions with new arguments
    toolbox.unregister("evaluate")
    toolbox.unregister("mutate")

    if strat_name == "reinit":
        toolbox.unregister("initDelta")
        toolbox.unregister("population")

    # Reset the partial clust counter to ceate new base clusters
    classes.PartialClust.id_value = count()

    # Save old genotype length
    relev_links_len_old = relev_links_len

    # Reduce delta by flat value or multiple of the square root if using that
    if isinstance(delta_val, int):
        delta_val -= delta_reduce
    else:
        delta_val -= (100*delta_reduce*np.sqrt(classes.Dataset.num_examples))/classes.Dataset.num_examples

    print(f"Adaptive delta engaged at gen {gen}! Going down to delta = {delta_val}")

    # Re-do the relevant precomputation
    relev_links_len, reduced_clust_nums = delta_precomp(delta_val, mst_genotype, int_links_indices)
    newly_unfixed_indices = int_links_indices[relev_links_len_old:relev_links_len]

    # Extend the individuals to the new genotype length
    for indiv in pop:
        indiv.extend([mst_genotype[i] for i in newly_unfixed_indices])

    # The evaluation function is the same for all (in current config)
    toolbox.register(
        "evaluate", 
        objectives.evalMOCK, 
        part_clust = classes.PartialClust.part_clust, 
        reduced_clust_nums = reduced_clust_nums, 
        conn_array = classes.PartialClust.conn_array, 
        max_conn = classes.PartialClust.max_conn, 
        num_examples = classes.Dataset.num_examples, 
        data_dict = data_dict, 
        cnn_pairs = classes.PartialClust.cnn_pairs, 
        base_members = classes.PartialClust.base_members, 
        base_centres = classes.PartialClust.base_centres
        )

    # Different mutation operators for different strategies
    if strat_name == "hypermutall":
        toolbox.register(
            "mutate", 
            operators.neighbourHyperMutation_all, 
            MUTPB = 1.0, 
            gen_length = relev_links_len, 
            argsortdists = argsortdists, 
            L = L, 
            int_links_indices = int_links_indices, 
            nn_rankings = nn_rankings, 
            hyper_mut = 500
            )

    elif strat_name == "hypermutspec":
        toolbox.register(
            "mutate", 
            operators.neighbourHyperMutation_spec, 
            MUTPB = 1.0, 
            gen_length = relev_links_len, 
            argsortdists = argsortdists, 
            L = L, 
            int_links_indices = int_links_indices, 
            nn_rankings = nn_rankings, 
            hyper_mut = 500, 
            new_genes = newly_unfixed_indices
            )

    elif strat_name == "fairmut":
        toolbox.register(
            "mutateFair", 
            operators.neighbourFairMutation, 
            MUTPB = 1.0, 
            gen_length = relev_links_len, 
            argsortdists = argsortdists, 
            L = L, 
            int_links_indices = int_links_indices, 
            nn_rankings = nn_rankings, 
            raised_mut = 50
            )
    
    elif strat_name == "reinit"       
        toolbox.register("initDelta", initialisation.initDeltaMOCKadapt, classes.Dataset.k_user, num_indivs, mst_genotype, int_links_indices, relev_links_len, argsortdists, L)
        toolbox.register("population", tools.initIterate, list, toolbox.initDelta)

    else:
        toolbox.register(
            "mutate", operators.neighbourMutation, 
            MUTPB = 1.0, 
            gen_length = relev_links_len, 
            argsortdists = argsortdists, 
            L = L, 
            int_links_indices = int_links_indices, 
            nn_rankings = nn_rankings
            )

    return pop, toolbox, delta_val, relev_links_len, reduced_clust_nums

# def main(**kwargs):
def main(data, data_dict, delta_val, HV_ref, argsortdists, nn_rankings, mst_genotype, int_links_indices, L, num_indivs, num_gens, delta_reduce):
    relev_links_len, reduced_clust_nums = delta_precomp(delta_val, mst_genotype, int_links_indices)

    toolbox = create_base_toolbox(num_indivs, mst_genotype, int_links_indices, relev_links_len, argsortdists, L, data_dict, nn_rankings)

    # Possibly try to modify this
    # Even consider using the with pool to limit no. pools
    # Can compare time
    pool = multiprocessing.Pool(processes = multiprocessing.cpu_count())
    toolbox.register("map", pool.map, chunksize=20)

    pop, VAR_init, CNN_init = initial_setup(toolbox)

    if HV_ref == None:
        HV_ref = calc_HV_ref(pop)

    # Check that the HV_ref reference point is valid
    if classes.PartialClust.max_conn >= HV_ref[1]:
        raise ValueError(f"Max CNN value ({classes.PartialClust.max_conn}) has exceeded that set for HV reference point ({HV_ref[1]}); HV values may be unreliable")

    if adapt_delta:
        window_size = 3 # Moving average of HV gradients to look at
        block_trigger_gens = 10 # Number of gens to wait between triggers
        adapt_gens = [0]
        max_adapts = 5 # Maximum number of adaptations allowed
        delta_reduce = 1 # Amount to reduce delta by
        ref_grad = None

    for gen in range(1, num_gens):
        # Select the right generation for the strategy employed
        if adapt_delta:
            if strat_name == "reinit":
                if adapt_gens[-1] == gen-1 and gen != 1:
                    pop, HV = reinit_generation(pop, toolbox, HV, HV_ref)
                else:
                    pop, HV = generation(pop, toolbox, HV, HV_ref)

        else:
            pop, HV = generation(pop, toolbox, HV, HV_ref)

        if adapt_delta:
            trigger_bool, ref_grad = check_trigger(delta_val, gen, adapt_gens, max_adapts, block_trigger_gens, HV, window_size, ref_grad)
            
            if trigger_bool:
                adapt_gens.append(gen)

                pop, toolbox, delta_val, relev_links_len, reduced_clust_nums = adaptive_delta_trigger(pop, gen, strat_name, delta_val, toolbox, delta_reduce, relev_links_len, mst_genotype, int_links_indices, num_indivs, argsortdists, L, data_dict, nn_rankings)

    pool.close()
    pool.join()

    classes.PartialClust.id_value = count()

    return pop, HV, HV_ref, int_links_indices, relev_links_len, adapt_gens