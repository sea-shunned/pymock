import precompute
import initialisation
import objectives
import operators
import classes

import numpy as np
from itertools import count
import random

from deap import base, creator, tools
from deap.benchmarks.tools import hypervolume

# Run outside of multiprocessing scope
# Can consieer trying to move this, though we just run it once anyway so eh
creator.create("Fitness", base.Fitness, weights=(-1.0, -1.0)) #(VAR, CNN)
creator.create("Individual", list, fitness=creator.Fitness, fairmut=None)

# Need to abstract this out into the run_mock file
def delta_precomp(data, data_dict, argsortdists, L, delta_val, mst_genotype, int_links_indices):
    """
    Do the precomputation specific to the delta value for that dataset
    """

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
def create_base_toolbox(num_indivs, mst_genotype, int_links_indices, relev_links_len, argsortdists, L, data_dict, nn_rankings, reduced_clust_nums):
    """
    Create the toolbox object used by the DEAP package, and register our relevant functions
    """
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

def initial_setup(toolbox, HV, HV_ref):
    """
    Do MOCK's initialisation and evaluate this initial population
    """

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
    fitnesses = [toolbox.evaluate(indiv) for indiv in pop]
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit	
        VAR_init.append(fit[0])
        CNN_init.append(fit[1])
    
    print("Max initial values:", max(VAR_init), max(CNN_init))

    # This is just to assign the crowding distance to the individuals, no actual selection is done
    pop = toolbox.select(pop, len(pop))

    # Add the hypervolume for the first generation to the list
    HV.append(hypervolume(pop, HV_ref))

    return pop, HV, VAR_init, CNN_init


def check_hv_violation(pop, hv_ref):
    # Check VAR
    if np.max([ind.fitness.values[0] for ind in pop]) > hv_ref[0]:
        raise ValueError("Intracluster variance has exceeded hv reference point")
    # Check CNN
    if np.max([ind.fitness.values[1] for ind in pop]) > hv_ref[1]:
        raise ValueError("Connectivity has exceeded hv reference point")


def generation(pop, toolbox, HV, HV_ref, num_indivs):
    """
    Perform a single generation of MOCK
    """

    # Shuffle the population
    random.shuffle(pop)

    # Select and clone the offspring (to allow modification)
    offspring = tools.selTournamentDCD(pop, len(pop))
    offspring = [toolbox.clone(ind) for ind in offspring]

    # Tournament
    for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
        # Crossover
        toolbox.mate(ind1, ind2)

        # Mutation
        toolbox.mutate(ind1)
        toolbox.mutate(ind2)

    # Evaluate the offspring
    fitnesses = [toolbox.evaluate(indiv) for indiv in offspring]
    for ind, fit in zip(offspring, fitnesses):
        ind.fitness.values = fit

    # Select from the current population and new offspring
    pop = toolbox.select(pop + offspring, num_indivs)

    # Add the hypervolume for this generation to the list
    HV.append(hypervolume(pop, HV_ref))

    return pop, HV

def generation_reinit(pop, toolbox, HV, HV_ref, num_indivs):
    """
    Perform a generation of MOCK for the reinitialisation strategy when triggered
    """

    # Shuffle the population
    random.shuffle(pop)

    # Generate the offspring from the initialisation routine
    offspring = toolbox.population()
    for index, indiv in enumerate(offspring):
        indiv = creator.Individual(indiv)
        offspring[index] = indiv

    # Evaluate the offspring
    fitnesses = [toolbox.evaluate(indiv) for indiv in offspring]
    for ind, fit in zip(offspring, fitnesses):
        ind.fitness.values = fit

    # Select from the current population and new offspring
    pop = toolbox.select(pop + offspring, num_indivs)

    # Add the hypervolume for this generation to the list
    HV.append(hypervolume(pop, HV_ref))

    return pop, HV 

def calc_hv_ref(VAR_init):
    return [np.ceil(np.max(VAR_init)*1.5), np.ceil(classes.PartialClust.max_conn+1)]

def check_trigger(delta_val, gen, adapt_gens, max_adapts, block_trigger_gens, HV, window_size, ref_grad):
    # Only trigger if delta is above zero
    if delta_val > 0:
        # Only trigger if we haven't reached our maximum limit
        if len(adapt_gens) < max_adapts:
            # Only trigger if we're outside an exploration period
            if gen >= (adapt_gens[-1] + block_trigger_gens):
                # Calculate a new reference gradient at the end of exploration
                if gen == (adapt_gens[-1] + block_trigger_gens):
                    ref_grad = (HV[-1] - HV[adapt_gens[-1]]) / len(HV)
                    print("Reference gradient:", ref_grad, "at gen", gen)
                    return False, ref_grad

                # Calculate the current moving average gradient
                curr_grad = (HV[-1] - HV[-(window_size+1)]) / window_size

                # If our gradient is much less than the reference, the search has slowed
                # So trigger a change in delta
                if curr_grad <= 0.1 * ref_grad:
                    return True, ref_grad

    return False, ref_grad

def adaptive_delta_trigger(pop, gen, strat_name, delta_val, toolbox, delta_reduce, relev_links_len, mst_genotype, int_links_indices, num_indivs, argsortdists, L, data_dict, nn_rankings, reduced_clust_nums, data):
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

    # Ensure delta doesn't go below zero
    if delta_val < 0:
        delta_val = 0

    print(f"Adaptive delta engaged at gen {gen}! Going down to delta = {delta_val}")

    # Re-do the relevant precomputation
    relev_links_len, reduced_clust_nums = delta_precomp(data, data_dict, argsortdists, L, delta_val, mst_genotype, int_links_indices)
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
    
    elif strat_name == "reinit":
        toolbox.register("initDelta", initialisation.initDeltaMOCKadapt, classes.Dataset.k_user, num_indivs, mst_genotype, int_links_indices, relev_links_len, argsortdists, L)
        
        toolbox.register("population", tools.initIterate, list, toolbox.initDelta)
        
        toolbox.register(
            "mutate", operators.neighbourMutation, 
            MUTPB = 1.0, 
            gen_length = relev_links_len, 
            argsortdists = argsortdists, 
            L = L, 
            int_links_indices = int_links_indices, 
            nn_rankings = nn_rankings
            )

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


def select_generation_strategy(
    pop, toolbox, HV, HV_ref, num_indivs, gen, 
    adapt_delta, adapt_gens, strat_name, relev_links_len, 
    argsortdists, L, int_links_indices, nn_rankings):
    if adapt_delta:
        if strat_name == "reinit":
            if adapt_gens[-1] == gen-1 and gen != 1:
                pop, HV = generation_reinit(pop, toolbox, HV, HV_ref, num_indivs)
            else:
                pop, HV = generation(pop, toolbox, HV, HV_ref, num_indivs)
        
        else:
            pop, HV = generation(pop, toolbox, HV, HV_ref, num_indivs)

        if "hypermut" in strat_name:
            if adapt_gens[-1] == gen-1 and gen != 1:
                toolbox.unregister("mutate")
                toolbox.register(
                    "mutate", operators.neighbourMutation, 
                    MUTPB = 1.0, 
                    gen_length = relev_links_len, 
                    argsortdists = argsortdists, 
                    L = L, 
                    int_links_indices = int_links_indices, 
                    nn_rankings = nn_rankings
                    )

    else:
        pop, HV = generation(pop, toolbox, HV, HV_ref, num_indivs)

    return pop, HV, toolbox


def runMOCK(
    data, data_dict, delta_val, hv_ref, argsortdists, 
    nn_rankings, mst_genotype, int_links_indices, L, 
    num_indivs, num_gens, delta_reduce, strat_name, adapt_delta, relev_links_len, reduced_clust_nums, seed_num
    ):
    """
    Run MOCK with specified inputs
    
    Arguments:
        data {np.array} -- array of the data
        data_dict {dict} -- dictionary of objects for each data point
        delta_val {int/float} -- MOCK delta value
        hv_ref {list} -- reference point for hypervolume calculation
        argsortdists {np.array} -- distance array of data argsorted
        nn_rankings {np.array} -- Neareast neighbour rankings for each data point
        mst_genotype {list} -- The genotype of the MST
        int_links_indices {list} -- Indices for the interesting links
        L {int} -- MOCK neighbourhood parameter
        num_indivs {int} -- Number of individuals in population
        num_gens {int} -- Number of generations
        delta_reduce {int} -- Amount to reduce delta by for adaptation
        strat_name {str} -- Name of strategy being run
        adapt_delta {bool} -- If delta should be adapted
        relev_links_len {int} -- Length/number of relevant links (i.e. length of reduced genotype)
        reduced_clust_nums {list} -- List of the cluster id numbers for the base clusters/components that are available in the search
    
    Returns:
        [type] -- [description]
    
    Returns:
        relev_links_len -- Length of the reduced/relevant genotype
        reduced_clust_nums -- Number of base clusters
    """

    random.seed(seed_num)

    # Initialise local varibles
    hv = []
    adapt_gens = None
    if adapt_delta:
        window_size = 3 # Moving average of hv gradients to look at
        block_trigger_gens = 10 # Number of gens to wait between triggers
        adapt_gens = [0]
        max_adapts = 5 # Maximum number of adaptations allowed
        delta_reduce = 1 # Amount to reduce delta by
        ref_grad = None # Reference gradient for hv trigger

    # Create the DEAP toolbox
    toolbox = create_base_toolbox(
        num_indivs, mst_genotype, int_links_indices, relev_links_len, argsortdists, L, data_dict, nn_rankings, reduced_clust_nums)

    # Create the initial population
    pop, hv, VAR_init, CNN_init = initial_setup(toolbox, hv, hv_ref)

    # Check that the initial population is within the hv reference point
    # check_hv_violation(pop, hv_ref)

    # Calculate a suitable ref point if not provided
    if hv_ref == None:
        hv_ref = calc_hv_ref(pop)
        # raise ValueError("No hv reference point supplied")

    # Check that the hv_ref reference point is valid
    if classes.PartialClust.max_conn >= hv_ref[1]:
        raise ValueError(f"Max CNN value ({classes.PartialClust.max_conn}) has exceeded that set for hv reference point ({hv_ref[1]}); hv values may be unreliable")

    # Go through each generation
    for gen in range(1, num_gens):
        # Select the right type of generation for this strategy and generation
        pop, hv, toolbox = select_generation_strategy(pop, toolbox, hv, hv_ref, num_indivs, gen, adapt_delta, adapt_gens, strat_name, relev_links_len, argsortdists, L, int_links_indices, nn_rankings)

        # Check if delta should be changed
        if adapt_delta:
            trigger_bool, ref_grad = check_trigger(delta_val, gen, adapt_gens, max_adapts, block_trigger_gens, hv, window_size, ref_grad)
            
            # Execute changes if triggered
            if trigger_bool:
                adapt_gens.append(gen)

                # Perform the trigger as specified by strat_name
                pop, toolbox, delta_val, relev_links_len, reduced_clust_nums = adaptive_delta_trigger(
                    pop, gen, strat_name, delta_val, toolbox, delta_reduce, relev_links_len, mst_genotype, int_links_indices, num_indivs, argsortdists, L, data_dict, nn_rankings, reduced_clust_nums, data)

    # Reset the ID count for the base clusters
    # classes.PartialClust.id_value = count()

    return pop, hv, hv_ref, int_links_indices, relev_links_len, adapt_gens

# add a main func here if we just want to run this thing once
# a main func is used within a script only, a main should not be imported