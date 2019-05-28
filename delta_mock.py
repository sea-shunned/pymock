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
        seed_num, data_dict, hv_ref, argsortdists,
        nn_rankings, L, num_indivs,
        num_gens, mut_meth_params
    ):
    """
    Run MOCK with specified inputs
    
    Arguments:
        data_dict {dict} -- dictionary of objects for each data point
        hv_ref {list} -- reference point for hypervolume calculation
        argsortdists {np.array} -- distance array of data argsorted
        nn_rankings {np.array} -- Neareast neighbour rankings for each data point
        L {int} -- MOCK neighbourhood parameter
        num_indivs {int} -- Number of individuals in population
        num_gens {int} -- Number of generations
        reduced_clust_nums {list} -- List of the cluster id numbers for the base clusters/components that are available in the search
    
    Returns:
        pop [type] -- Final population
        hv [list] -- Hypervolume values
        hv_ref [list] -- the nadir point (return in case we want to check)
        interest_indices [list] -- Final indices for interesting links
        reduced_length [int] -- Final length of genotype
        time_taken [float] -- The time taken for this run
    """
    # Set the seed
    random.seed(seed_num)
    np.random.seed(seed_num) # Currently unused, should switch to in future
    print(f"Seed number: {seed_num}")

    start_time = time.time()
    # Initialise local varibles
    # This can be abstracted out in future
    hv = []

    # Create the DEAP toolbox
    toolbox = create_base_toolbox(
        num_indivs, argsortdists, L, data_dict, nn_rankings, mut_meth_params
    )

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
        # Perform a single generation
        pop, hv = generation(pop, toolbox, hv, hv_ref, num_indivs)
    # Measure the time taken
    time_taken = time.time() - start_time
    return pop, hv, hv_ref, MOCKGenotype.interest_indices, MOCKGenotype.reduced_length, time_taken
