# Standard
import random
import time
# External
import numpy as np
from deap import base, creator, tools
from deap.benchmarks.tools import hypervolume
from tqdm import tqdm
# Own
import precompute
import initialisation
import objectives
import operators
from classes import Datapoint, MOCKGenotype, PartialClust
# New
# import no_precomp_objectives

# Run outside of multiprocessing scope
# Can consider trying to move this, though we just run it once anyway so eh
creator.create("Fitness", base.Fitness, weights=(-1.0, -1.0)) #(VAR, CNN)
creator.create("Individual", list, fitness=creator.Fitness, fairmut=None)


# Consider trying to integrate the use of **kwargs here
def create_base_toolbox(num_indvs, argsortdists, L, data_dict,
                        nn_rankings, mut_meth_params, domain, min_sr, init_sr, min_delta, init_delta, max_delta,
                        delta_mutation, delta_precision, delta_mutpb, delta_sigma,
                        delta_sigma_as_perct, delta_inverse, crossover, flexible_limits,
                        squash):
    """
    Create the toolbox object used by the DEAP package, and register our relevant functions
    """
    toolbox = base.Toolbox()

    # Register the individual creator
    toolbox.register(
        "individual",
        MOCKGenotype.delta_individual,
        icls=creator.Individual,
        min_delta=init_delta,
        max_delta=max_delta,
        precision=delta_precision
    )

    # Register the initialisation function
    toolbox.register(
        "initDelta",
        initialisation.init_uniformly_distributed_population,
        num_indvs=num_indvs,
        k_user=Datapoint.k_user,
        min_delta=init_delta,
        max_delta=max_delta,
        argsortdists=argsortdists,
        L=L,
        indiv_creator=toolbox.individual,
        domain=domain,
        precision=delta_precision
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
        base_centres=PartialClust.base_centres,
        toolbox=toolbox
    )

    # Register the crossover function
    if crossover == 'uniform':
        toolbox.register(
            "mate",
            operators.uniform_xover,
            cxpb=1.0
        )
    elif crossover is None:
        toolbox.register(
            "mate",
            operators.no_xover
        )

    # Register the mutation function
    if mut_meth_params['mut_method'] == "original":
        toolbox.register(
            "mutate", operators.neighbour_mut,
            MUTPB=1.0,
            argsortdists=argsortdists,
            L=L,
            interest_indices=MOCKGenotype.interest_indices,
            nn_rankings=nn_rankings
        )
    elif mut_meth_params['mut_method'] == "centroid":
        toolbox.register(
            "mutate", operators.comp_centroid_mut,
            MUTPB=1.0,
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
            interest_indices=MOCKGenotype.interest_indices,
            nn_rankings=nn_rankings,
            component_nns=mut_meth_params['component_nns'],
            data_dict=data_dict
        )

    # Register delta mutation
    if delta_mutation == 'gauss':
        toolbox.register(
            "mutate_delta", operators.gaussian_mutation_delta,
            sigma=delta_sigma,
            mutpb=delta_mutpb,
            precision=delta_precision,
            min_sr=min_sr,
            init_sr=init_sr,
            min_delta=min_delta,
            max_delta=max_delta,
            sigma_perct=delta_sigma_as_perct,
            inverse=delta_inverse,
            flexible_limits=flexible_limits,
            squash=squash
        )
    elif delta_mutation == 'uniform':
        toolbox.register(
            "mutate_delta", operators.uniform_mutation_delta,
            spread=delta_sigma,
            mutpb=delta_mutpb,
            precision=delta_precision,
            min_sr=min_sr,
            init_sr=init_sr,
            min_delta=min_delta,
            max_delta=max_delta,
            flexible_limits=flexible_limits,
            squash=squash
        )
    elif delta_mutation == 'random':
        toolbox.register(
            "mutate_delta", operators.random_delta,
            precision=delta_precision,
            min_sr=min_sr,
            init_sr=init_sr,
            max_delta=max_delta
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
    n = len(pop)
    init_pop = toolbox.initDelta()
    # Convert each individual of class list to class deap.creator.Individual
    # Easier than modifying population function
    for index, indiv in enumerate(init_pop):
        pop[index] = indiv

    # Lists to capture the initial population fitness (if desired)
    VAR_init = []
    CNN_init = []

    # Evaluate the initial po
    for ind in pop:
        var, cnn = toolbox.evaluate(ind)
        ind.fitness.values = (var, cnn)
        VAR_init.append(var)
        CNN_init.append(cnn)
    # print("Max initial values:", max(VAR_init), max(CNN_init))

    # This is just to assign the crowding distance to the individuals, no actual selection is done
    pop = toolbox.select(pop, n)

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


def generation(pop, toolbox, HV, HV_ref, num_indvs, init_delta, gen_n=0):
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
        toolbox.mate(ind1, ind2)

        # Mutate delta
        toolbox.mutate_delta(ind1, init_delta=init_delta, gen_n=gen_n)
        toolbox.mutate_delta(ind2, init_delta=init_delta, gen_n=gen_n)

        # Mutation
        toolbox.mutate(ind1)
        toolbox.mutate(ind2)

    # Evaluate the offspring
    for ind in offspring:
        var, cnn = toolbox.evaluate(ind)
        ind.fitness.values = (var, cnn)

    # Select from the current population and new offspring
    pop = toolbox.select(pop + offspring, num_indvs)

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
        seed_num, data, data_dict, hv_ref, argsortdists,
        nn_rankings, L, num_indvs,
        num_gens, mut_meth_params, domain, min_sr, init_sr, min_delta, max_delta, init_delta,
        delta_mutation, delta_precision, delta_mutpb, delta_sigma,
        delta_sigma_as_perct, delta_inverse, crossover, flexible_limits, squash,
        gens_step, stair_limits, run_number, save_history, verbose
    ):
    """
    Run MOCK with specified inputs
    
    Arguments:
        data_dict {dict} -- dictionary of objects for each data point
        hv_ref {list} -- reference point for hypervolume calculation
        argsortdists {np.array} -- distance array of data argsorted
        nn_rankings {np.array} -- Neareast neighbour rankings for each data point
        L {int} -- MOCK neighbourhood parameter
        num_indvs {int} -- Number of individuals in population
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
    if gens_step is None:
        gens_step = 0.1

    # Set the seed
    random.seed(seed_num)
    np.random.seed(seed_num)  # Currently unused, should switch to in future

    start_time = time.time()
    # Initialise local varibles
    # This can be abstracted out in future
    hv = []

    # Create the DEAP toolbox
    toolbox_params = {
        'num_indvs': num_indvs, 'argsortdists': argsortdists, 'L': L, 'data_dict': data_dict,
        'nn_rankings': nn_rankings, 'mut_meth_params': mut_meth_params, 'domain': domain, 'min_sr': min_sr,
        'init_sr': init_sr, 'init_delta': init_delta, 'min_delta':min_delta, 'max_delta': max_delta,
        'delta_mutation': delta_mutation, 'delta_precision': delta_precision, 'delta_mutpb': delta_mutpb,
        'delta_sigma': delta_sigma, 'delta_sigma_as_perct': delta_sigma_as_perct, 'delta_inverse': delta_inverse,
        'crossover': crossover, 'flexible_limits': flexible_limits, 'squash': squash
    }
    toolbox = create_base_toolbox(**toolbox_params)

    # Create the initial population
    pop, hv, VAR_init, CNN_init = initial_setup(toolbox, hv, hv_ref)

    # Check that the initial population is within the hv reference point
    # check_hv_violation(pop, hv_ref)

    # # Check that the hv_ref reference point is valid
    if PartialClust.max_cnn >= hv_ref[1]:
        raise ValueError(f"Max CNN value ({PartialClust.max_cnn}) has exceeded that set for hv reference point ({hv_ref[1]}); hv values may be unreliable")

    # Go through each generation
    all_pop = []
    if verbose:
        pbar = tqdm(total=num_gens)
        pbar.set_description(f"Run {run_number}")
    for gen in range(1, num_gens+1):
        # Calculate new init_delta:
        if gen % gens_step == 0:
            init_delta -= stair_limits

        if gen == int(flexible_limits) + 1:
            # Recalculate precomputation
            MOCKGenotype.setup_genotype_vars(min_delta, data, data_dict, argsortdists, L, domain=domain, max_sr=min_sr)
            toolbox = create_base_toolbox(**toolbox_params)

        # Perform a single generation
        pop, hv = generation(pop, toolbox, hv, hv_ref, num_indvs, init_delta, gen)

        # Save the results
        if save_history:
            all_pop.append(pop)

        if verbose:
            pbar.update(1)

    if verbose:
        pbar.close()

    # Measure the time taken
    time_taken = time.time() - start_time

    if not save_history:
        all_pop.append(pop)

    return all_pop, hv, hv_ref, MOCKGenotype.interest_indices, MOCKGenotype.reduced_length, time_taken
