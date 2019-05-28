# Standard
import random
import time
from itertools import count
# External
import numpy as np
from deap import base, creator, tools
from deap.benchmarks.tools import hypervolume
# Own
import precompute
import initialisation
import objectives
import operators
from classes import Datapoint, MOCKGenotype, PartialClust

# Run outside of multiprocessing scope
# Can consider trying to move this, though we just run it once anyway so eh
creator.create("Fitness", base.Fitness, weights=(-1.0, -1.0)) #(VAR, CNN)
creator.create("Individual", list, fitness=creator.Fitness, fairmut=None)

# Consider trying to integrate the use of **kwargs here
def create_base_toolbox(num_indivs, argsortdists, L, data_dict,
                        nn_rankings, mut_meth_params):
    """
    Create the toolbox object used by the DEAP package, and register our relevant functions
    """
    toolbox = base.Toolbox()

    # Register the initialisation function
    toolbox.register(
        "initDelta",
        initialisation.init_deltamock,
        k_user=Datapoint.k_user,
        num_indivs=num_indivs,
        argsortdists=argsortdists,
        L=L
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
        objectives.eval_mock,
        comp_dict=PartialClust.comp_dict,
        reduced_clust_nums=MOCKGenotype.reduced_cluster_nums,
        cnn_array=PartialClust.cnn_array,
        max_cnn=PartialClust.max_cnn,
        num_examples=Datapoint.num_examples,
        data_dict=data_dict,
        cnn_pairs=PartialClust.cnn_pairs,
        base_members=PartialClust.base_members,
        base_centres=PartialClust.base_centres
    )

    # Register the crossover function
    toolbox.register(
        "mate",
        operators.uniform_xover,
        cxpb=1.0
    )

    # Register the mutation function
    if mut_meth_params['mut_method'] == "original":
        toolbox.register(
            "mutate", operators.neighbour_mut,
            MUTPB=1.0,
            gen_length=MOCKGenotype.reduced_length,
            argsortdists=argsortdists,
            L=L,
            interest_indices=MOCKGenotype.interest_indices,
            nn_rankings=nn_rankings
        )

    elif mut_meth_params['mut_method'] == "centroid":
        toolbox.register(
            "mutate", operators.comp_centroid_mut,
            MUTPB=1.0,
            gen_length=MOCKGenotype.reduced_length,
            argsortdists_cen=mut_meth_params['argsortdists_cen'],
            L_comp=mut_meth_params['L_comp'],
            interest_indices=MOCKGenotype.interest_indices,
            nn_rankings_cen=mut_meth_params['nn_rankings_cen'],
            data_dict=data_dict
        )
    
    elif mut_meth_params['mut_method'] == "neighbour":
        toolbox.register(
            "mutate", operators.neighbour_comp_mut,
            MUTPB=1.0,
            gen_length=MOCKGenotype.reduced_length,
            interest_indices=MOCKGenotype.interest_indices,
            nn_rankings=nn_rankings,
            component_nns=mut_meth_params['component_nns'],
            data_dict=data_dict
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
    
    # print("Max initial values:", max(VAR_init), max(CNN_init))

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
                    # print("Reference gradient:", ref_grad, "at gen", gen)
                    return False, ref_grad

                # Calculate the current moving average gradient
                curr_grad = (HV[-1] - HV[-(window_size+1)]) / window_size

                # If our gradient is much less than the reference, the search has slowed
                # So trigger a change in delta
                if curr_grad <= 0.1 * ref_grad:
                    return True, ref_grad

    return False, ref_grad

def adaptive_delta_trigger(pop, gen, strategy, delta_val, toolbox,
    delta_reduce, num_indivs, argsortdists, L, data_dict, nn_rankings, data):
    # Need to re-register the functions with new arguments
    toolbox.unregister("evaluate")
    toolbox.unregister("mutate")

    # Reset the partial clust counter to ceate new base clusters
    PartialClust.id_value = count()

    # Save old genotype length
    reduced_len_old = MOCKGenotype.reduced_length

    # Reduce delta by flat value or multiple of the square root if using that
    if isinstance(delta_val, int):
        MOCKGenotype.delta_val -= delta_reduce
    else:
        MOCKGenotype.delta_val -= (100*delta_reduce*np.sqrt(Datapoint.num_examples))/Datapoint.num_examples

    # Ensure delta doesn't go below zero
    if MOCKGenotype.delta_val < 0:
        MOCKGenotype.delta_val = 0

    print(f"Adaptive delta engaged at gen {gen}! Going down to delta = {MOCKGenotype.delta_val}")

    # Re-do the relevant precomputation # Need to do in new MOCKGenotype
    # reduced_length, reduced_clust_nums = delta_precomp(data, data_dict, argsortdists, L, delta_val, mst_genotype, interest_indices) # FIX

    MOCKGenotype.setup_genotype_vars()
    PartialClust.partial_clusts(data, data_dict, argsortdists, L)
    MOCKGenotype.calc_reduced_clusts(data_dict)    
    
    # newly_unfixed_indices = interest_indices[reduced_len_old:reduced_length]
    newly_unfixed_indices = MOCKGenotype.interest_indices[reduced_len_old:MOCKGenotype.reduced_length]

    # Extend the individuals to the new genotype length
    for indiv in pop:
        indiv.extend([MOCKGenotype.mst_genotype[i] for i in newly_unfixed_indices])

    # The evaluation function is the same for all (in current config)
    toolbox.register(
        "evaluate",
        objectives.eval_mock,
        comp_dict=PartialClust.comp_dict,
        reduced_clust_nums=MOCKGenotype.reduced_cluster_nums,
        cnn_array=PartialClust.cnn_array,
        max_cnn=PartialClust.max_cnn,
        num_examples=Datapoint.num_examples,
        data_dict=data_dict,
        cnn_pairs=PartialClust.cnn_pairs,
        base_members=PartialClust.base_members,
        base_centres=PartialClust.base_centres
    )

    # Different mutation operators for different strategies
    if strategy == "hypermutall":
        toolbox.register(
            "mutate",
            operators.neighbourHyperMutation_all,
            MUTPB=1.0,
            gen_length=MOCKGenotype.reduced_length,
            argsortdists=argsortdists,
            L=L,
            interest_indices=MOCKGenotype.interest_indices,
            nn_rankings=nn_rankings,
            hyper_mut=500
        )

    elif strategy == "hypermutspec":
        toolbox.register(
            "mutate",
            operators.neighbourHyperMutation_spec,
            MUTPB=1.0,
            gen_length=MOCKGenotype.reduced_length,
            argsortdists=argsortdists,
            L=L,
            interest_indices=MOCKGenotype.interest_indices,
            nn_rankings=nn_rankings,
            hyper_mut=500,
            new_genes=newly_unfixed_indices
        )

    elif strategy == "fairmut":
        toolbox.register(
            "mutateFair",
            operators.neighbourFairMutation,
            MUTPB=1.0,
            gen_length=MOCKGenotype.reduced_length,
            argsortdists=argsortdists,
            L=L,
            interest_indices=MOCKGenotype.interest_indices,
            nn_rankings=nn_rankings,
            raised_mut=50
        )

    elif strategy == "reinit":
        # Unregister population functions first
        toolbox.unregister("initDelta")
        toolbox.unregister("population")

        toolbox.register(
            "initDelta",
            initialisation.init_deltamockadapt,
            Datapoint.k_user,
            num_indivs,
            argsortdists,
            L
        )
        
        toolbox.register(
            "population", tools.initIterate, list, toolbox.initDelta
        )
        
        toolbox.register(
            "mutate", operators.neighbour_mut,
            MUTPB=1.0,
            gen_length=MOCKGenotype.reduced_length,
            argsortdists=argsortdists,
            L=L,
            interest_indices=MOCKGenotype.interest_indices,
            nn_rankings=nn_rankings
        )

    else:
        toolbox.register(
            "mutate", operators.neighbour_mut,
            MUTPB=1.0,
            gen_length=MOCKGenotype.reduced_length,
            argsortdists=argsortdists,
            L=L,
            interest_indices=MOCKGenotype.interest_indices,
            nn_rankings=nn_rankings
        )
    # Fix reduced_length here (need to look where returned)
    return pop, toolbox


def select_generation_strategy(
        pop, toolbox, HV, HV_ref, num_indivs, gen, 
        adapt_delta, adapt_gens, strategy,
        argsortdists, L, nn_rankings):
    # Check if we're adapting delta
    if adapt_delta:
        # If using the RO strategy
        if strategy == "reinit":
            # If we just triggered a change, generate offspring using strategy
            if adapt_gens[-1] == gen-1 and gen != 1:
                pop, HV = generation_reinit(pop, toolbox, HV, HV_ref, num_indivs)
            # Otherwise normal
            else:
                pop, HV = generation(pop, toolbox, HV, HV_ref, num_indivs)
        # All other strategies 
        else:
            pop, HV = generation(pop, toolbox, HV, HV_ref, num_indivs)

        # If using a hypermutation strategy, we only apply rate for 1 generation
        # So revert back to normal mutation operator
        if "hypermut" in strategy:
            if adapt_gens[-1] == gen-1 and gen != 1:
                toolbox.unregister("mutate")
                toolbox.register(
                    "mutate", operators.neighbour_mut,
                    MUTPB=1.0,
                    gen_length=MOCKGenotype.reduced_length,
                    argsortdists=argsortdists,
                    L=L,
                    interest_indices=MOCKGenotype.interest_indices,
                    nn_rankings=nn_rankings
                )
    # Non-adaptive
    else:
        pop, HV = generation(pop, toolbox, HV, HV_ref, num_indivs)
    return pop, HV, toolbox

def get_mutation_params(mut_method, mock_args, L_comp=None):
    if mut_method == "centroid":
        distarray_cen = precompute.compute_dists(
            PartialClust.base_centres, PartialClust.base_centres
        )
        mock_args['mut_meth_params'] = {
            'mut_method': "centroid",
            'argsortdists_cen': np.argsort(
                distarray_cen, kind='mergesort'
            ),
            'nn_rankings_cen': precompute.nn_rankings(
                distarray_cen, len(PartialClust.comp_dict)
            ),
            'L_comp': L_comp
        }                
    elif mut_method == "neighbour":
        mock_args['mut_meth_params'] = {
            'mut_method': "neighbour",
            'component_nns': precompute.component_nn(
                Datapoint.num_examples, mock_args['argsortdists'],
                mock_args['data_dict'], L_comp
            )
        }
    else:
        mock_args['mut_meth_params'] = {
            'mut_method': "original"
        }    
    return mock_args

def runMOCK(
    seed_num, data, data_dict, hv_ref, argsortdists,
    nn_rankings, L, num_indivs, num_gens, 
    strategy, adapt_delta, mut_meth_params
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
        interest_indices {list} -- Indices for the interesting links
        L {int} -- MOCK neighbourhood parameter
        num_indivs {int} -- Number of individuals in population
        num_gens {int} -- Number of generations
        delta_reduce {int} -- Amount to reduce delta by for adaptation
        strategy {str} -- Name of strategy being run
        adapt_delta {bool} -- If delta should be adapted
        reduced_clust_nums {list} -- List of the cluster id numbers for the base clusters/components that are available in the search
    
    Returns:
        pop [type] -- Final population
        hv [list] -- Hypervolume values
        hv_ref [list] -- the nadir point (return in case we want to check)
        interest_indices [list] -- Final indices for interesting links
        reduced_length [int] -- Final length of genotype
        adapt_gens [list] -- Generations where delta change was triggered
    """
    # Set the seed
    random.seed(seed_num)
    np.random.seed(seed_num) # Currently unused, should switch to in future
    print(f"Seed number: {seed_num}")

    start_time = time.time()
    # Initialise local varibles
    # This can be abstracted out in future
    hv = []
    adapt_gens = None
    # Set the parameters for adaptive delta mechanism
    if adapt_delta:
        window_size = 3 # Moving average of hv gradients to look at
        block_trigger_gens = 10 # Number of gens to wait between triggers
        adapt_gens = [0] # Which generations delta adapts at
        max_adapts = 5 # Maximum number of adaptations allowed
        delta_reduce = 1 # Amount to reduce delta by
        ref_grad = None # Reference gradient for hv trigger

    # Create the DEAP toolbox
    toolbox = create_base_toolbox(
        num_indivs, argsortdists, L, data_dict, nn_rankings, mut_meth_params)

    # Create the initial population
    pop, hv, VAR_init, CNN_init = initial_setup(toolbox, hv, hv_ref)

    # Check that the initial population is within the hv reference point
    # check_hv_violation(pop, hv_ref)

    # # Check that the hv_ref reference point is valid
    # if PartialClust.max_cnn >= hv_ref[1]:
    #     raise ValueError(f"Max CNN value ({PartialClust.max_cnn}) has exceeded that set for hv reference point ({hv_ref[1]}); hv values may be unreliable")

    # Go through each generation
    for gen in range(1, num_gens):
        # if gen % 10 == 0:
        #     print(f"Generation: {gen}")

        # Select the right type of generation for this strategy and generation
        pop, hv, toolbox = select_generation_strategy(
            pop, toolbox, hv, hv_ref, num_indivs, gen, adapt_delta,
            adapt_gens, strategy, argsortdists, L, nn_rankings
        )
        # Check if delta should be changed
        if adapt_delta:
            trigger_bool, ref_grad = check_trigger(
                MOCKGenotype.delta_val, gen, adapt_gens, max_adapts, block_trigger_gens, hv, window_size, ref_grad
            )
            # Execute changes if triggered
            if trigger_bool:
                adapt_gens.append(gen)
                # Perform the trigger as specified by strategy
                pop, toolbox = adaptive_delta_trigger(
                    pop, gen, strategy, MOCKGenotype.delta_val, 
                    toolbox, delta_reduce, num_indivs, argsortdists, L, data_dict, nn_rankings, data
                )
    time_taken = time.time() - start_time
    # Reset the ID count for the base clusters
    # PartialClust.id_value = count()
    return pop, hv, hv_ref, MOCKGenotype.interest_indices, MOCKGenotype.reduced_length, adapt_gens, time_taken

# add a main func here if we just want to run this thing once
# a main func is used within a script only, a main should not be imported