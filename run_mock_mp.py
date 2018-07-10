# Standard libraries
import os
import glob
import json
import random
import time
from functools import partial
from itertools import count
import multiprocessing
import numpy as np

# Own functions
from classes import Dataset, MOCKGenotype, PartialClust
import precompute
import evaluation
import objectives
import delta_mock_mp
from tests import validateResults

def load_data(use_real_data=False, synth_data_subset="*", real_data_subset="*"):
    """Get the file paths for all the data we're using
    
    Keyword Arguments:
        use_real_data {bool} -- The real data consumes a lot of memory, can choose to exclude it (default: {False})
        synth_data_subset {str} -- Allows for selecting specific datasets from the synthetic set (default: {"*"})
        real_data_subset {str} -- As above for the real (UKC) data (default: {"*"})
    
    Returns:
        [list] -- List of all the data files
        results_folder [str] -- Where the results should be saved
    """
    # Get the current directory path
    base_path = os.getcwd()

    data_folder = os.path.join(base_path, "data", "")
    results_folder = os.path.join(base_path, "results", "")

    synth_data_folder = os.path.join(data_folder, "synthetic_datasets", "")
    synth_data_files = glob.glob(synth_data_folder+synth_data_subset+".data")

    if use_real_data:
        real_data_folder = os.path.join(data_folder, "UKC_datasets", "")
        real_data_files = glob.glob(real_data_folder+real_data_subset+".txt")
    else:
        real_data_files = []
    
    return synth_data_files + real_data_files, results_folder


def prepare_data(file_path, L=10, num_indivs=100, num_gens=100, delta_reduce=1):
    """Prepare the dataset (precomputation). Default values are overwritten by config file
    
    Arguments:
        file_path {[type]} -- [description]
    
    Keyword Arguments:
        L {int} -- The neighbourhood hyperparameter (default: {10})
        num_indivs {int} -- Number of individuals (default: {100})
        num_gens {int} -- Number of generations (default: {100})
        delta_reduce {int} -- Amount (square root multiple) to reduce delta (default: {1})
    
    Returns:
        kwargs [dict] -- All the arguments we need to run MOCK
    """

    if "synthetic" in file_path:
        Dataset.data_name = file_path.split("/")[-1].split(".")[0][:-15]
    elif "UKC" in file_path:
        Dataset.data_name = file_path.split("/")[-1].split(".")[0]
    # Current data has header with metadata
    with open(file_path) as file:
        head = [int(next(file)[:-1]) for _ in range(4)]

    # Read the data in as an array
    # The skip_header is for the header info in this data specifically
    data = np.genfromtxt(file_path, delimiter="\t", skip_header=4)

    # Set the values for the data
    Dataset.num_examples = head[0] # Num examples
    Dataset.num_features = head[1] # Num features/dimensions
    Dataset.k_user = head[3] # Num real clusters

    # Do we have labels?
    if head[2] == 1:
        Dataset.labels = True
    else:
        Dataset.labels = False

    # Remove labels if present and create data_dict
    data, data_dict = Dataset.createDatasetGarza(data)    

    # Go through the precomputation specific to the dataset
    # Calculate distance array
    distarray = precompute.compDists(data, data)
    distarray = precompute.normaliseDistArray(distarray)
    argsortdists = np.argsort(distarray, kind='mergesort')
    
    # Calculate nearest neighbour rankings
    nn_rankings = precompute.nnRankings(distarray, Dataset.num_examples)
    
    # Calculate MST
    MOCKGenotype.mst_genotype = precompute.createMST(distarray)
    
    # Calculate DI values
    MOCKGenotype.degree_int = precompute.degreeInterest(MOCKGenotype.mst_genotype, L, nn_rankings, distarray)
    
    # Sort to get the indices of most to least interesting links
    MOCKGenotype.interest_links_indices()

    # Bundle all of the arguments together in a dict to pass to the function
    # This is in order of runMOCK() so that we can easily turn it into a partial func for multiprocessing
    kwargs = {
        "data": data,
        "data_dict": data_dict,
        "delta_val": None,
        "hv_ref": None,
        "argsortdists": argsortdists,
        "nn_rankings": nn_rankings,
        "mst_genotype": MOCKGenotype.mst_genotype,
        "interest_indices": MOCKGenotype.interest_indices,
        "L": L,
        "num_indivs": num_indivs,
        "num_gens": num_gens,
        "delta_reduce": delta_reduce,
        "strat_name": None,
        "adapt_delta": None,
        "reduced_length": None,
        "reduced_clust_nums": None
        # "seed_num": None
    }
    return kwargs


def create_seeds(NUM_RUNS, seed_file=None, save_new_seeds=True):
    """Create the seed numbers
    
    Arguments:
        NUM_RUNS {int} -- Number of runs
    
    Keyword Arguments:
        seed_file {str} -- File location if giving previous seeds (default: {None})
        save_new_seeds {bool} -- If we should save our new seeds to reproduce same experiment (default: {True})
    
    Raises:
        ValueError -- Error if we don't have enough seeds
    
    Returns:
        seed_list [list] -- List of the seed numbers
    """
    # Load seeds if present
    if seed_file is not None:
        params = load_config(seed_file)
        seed_list = params['seed_list']
    else:
        # Randomly generate seeds
        seed_list = [random.uniform(0, 1000) for i in range(NUM_RUNS)]

        # Save a new set of seeds for this set of experiments
        # (to ensure same start for each strategy)
        if save_new_seeds:
            import datetime
            seed_fname = "seed_list_"+str(datetime.date.today())+".json"
            with open(seed_fname, 'w') as out_file:
                json.dump(seed_list, out_file, indent=4)
                
    # Ensure we have enough seeds
    if len(seed_list) < NUM_RUNS:
        raise ValueError("Not enough seeds for number of runs")

    return seed_list


def load_config(config_path="mock_config.json"):
    """Load config file for MOCK
    
    Keyword Arguments:
        config_path {str} -- [description] (default: {"mock_config.json"})
    
    Returns:
        [type] -- [description]
    """
    try:
        with open(config_path) as json_file:
            params = json.load(json_file)
    except JSONDecodeError:
        print("Unable to load config file")
        raise
    return params

def calc_hv_ref(kwargs):
    """Calculates a hv reference/nadir point for use on all runs of that dataset
    
    Arguments:
        kwargs {dict} -- Arguments for MOCK
    
    Returns:
        [list] -- The nadir point
    """


    mst_reduced_genotype = [MOCKGenotype.mst_genotype[i] for i in MOCKGenotype.reduced_genotype_indices]

    # Calculate 
    chains, superclusts = objectives.clusterChains(
        mst_reduced_genotype, kwargs['data_dict'], PartialClust.part_clust, MOCKGenotype.reduced_cluster_nums
    )
    
    PartialClust.max_var = objectives.objVAR(
        chains, PartialClust.part_clust, PartialClust.base_members,
        PartialClust.base_centres, superclusts
    )

    # Need to divide by N as this is done in evalMOCK() not objVAR()
    PartialClust.max_var= PartialClust.max_var/Dataset.num_examples

    # Set reference point just outside max values to ensure no overlap
    hv_ref = [
        PartialClust.max_var*1.01,
        PartialClust.max_conn*1.01
    ]

    print("Max var:", PartialClust.max_var)

    PartialClust.id_value = count()

    return hv_ref

# Maybe this should be main
def run_mock(validate=False):
    # Load the data file paths
    data_file_paths, results_folder = load_data(
        synth_data_subset="tevc_20_60_9*")

    # Load general MOCK parameyers
    params = load_config(config_path="mock_config.json")

    # A longer genotype means a higher possible maximum for the CNN objective
    # By running the highest sr value first, we ensure that the HV_ref
    # is the same and appropriate for all runs
    sr_vals = sorted(params['sr_vals'], reverse=True)

    # Column names for fitness array, formatted for EAF R plot package
    fitness_cols = ["VAR", "CNN", "Run"]

    # Create the seed list or load existing one
    if validate:
        seed_list = create_seeds(params['NUM_RUNS'], seed_file="seed_list.json")
    else:
        seed_list = create_seeds(params['NUM_RUNS'])

    # Restrict seed_list to the actual number of runs that we need
    # Truncating like this allows us to know that the run numbers and order of seeds correspond
    seed_list = seed_list[:params['NUM_RUNS']]
    seed_list = [(i,) for i in seed_list]

    # Print the number of runs to get an idea of runtime (sort of)
    print("---------------------------")
    print("Number of MOCK Runs:")
    print(f"{params['NUM_RUNS']} runs")
    print(f"{len(params['strategies'])} strategy/-ies")
    print(f"{len(sr_vals)} delta values")
    print(f"{len(data_file_paths)} datasets")
    print(f"= {params['NUM_RUNS']*len(params['strategies'])*len(sr_vals)*len(data_file_paths)} runs")
    print("---------------------------")

    # Loop through the data to test
    for file_path in data_file_paths:
        print(f"Beginning precomputation for {file_path.split(os.sep)[-1]}")
        kwargs = prepare_data(file_path, params['L'], params['NUM_INDIVS'], params['NUM_GENS'])
        print("Precomputation complete")


        # Loop through the sr (square root) values
        for sr_val in sr_vals:
            # Calculate the delta value from the sr
            MOCKGenotype.calc_delta(sr_val)
            kwargs['delta_val'] = MOCKGenotype.delta_val
            
            print(f"Delta: {kwargs['delta_val']}")

            # Setup some of the variables for the genotype
            MOCKGenotype.setup_genotype_vars()

            # Set the nadir point if first run
            if kwargs['hv_ref'] is None:
                # To ensure compatible results
                if validate:
                    kwargs['hv_ref'] = [3.0, 1469.0]
                else:
                    kwargs['hv_ref'] = calc_hv_ref(kwargs, sr_vals)

            # Then need to check if we need to register anything new with kwargs
            # If we just add the class stuff to kwargs we don't have to make many other changes
            
            PartialClust.partial_clusts(kwargs["data"], kwargs["data_dict"], kwargs["argsortdists"], kwargs["L"])
            MOCKGenotype.calc_reduced_clusts(kwargs["data_dict"])

            # Loop through the strategies
            for strat_name in params['strategies']:
                kwargs['strat_name'] = strat_name
                if strat_name == "base":
                    kwargs['adapt_delta'] = False
                else:
                    kwargs['adapt_delta'] = True

                fitness_array = np.empty((params['NUM_INDIVS']*params['NUM_RUNS'], len(fitness_cols)))
                hv_array = np.empty((params['NUM_GENS'], params['NUM_RUNS']))
                ari_array = np.empty((params['NUM_INDIVS'], params['NUM_RUNS']))
                num_clusts_array = np.empty((params['NUM_INDIVS'], params['NUM_RUNS']))
                time_array = np.empty(params['NUM_RUNS'])
                delta_triggers = []

                mock_func = partial(delta_mock_mp.runMOCK, *list(kwargs.values()))

                print(f"{strat_name} starting...")
                # Measure the time taken for the runs
                start_time = time.time()
                # Send the function to a thread, each thread with a different seed
                with multiprocessing.Pool() as pool:
                    results = pool.starmap(mock_func, seed_list)
                end_time = time.time()
                print(f"{strat_name} done (took {end_time-start_time:.3f} secs) - collecting results...")

                for run_num, run_result in enumerate(results):
                    # Extract the population
                    pop = run_result[0]

                    # temp_res = sorted([ind.fitness.values for ind in pop], key=lambda var:var[1])
                    # print(temp_res[:2])
                    # print(temp_res[-2:])

                    # Extract hv list
                    hv = run_result[1]
                    # Extract final interesting links
                    final_interest_inds = run_result[3]
                    # Extract final genotype length
                    final_gen_len = run_result[4]
                    # Extract gens with delta trigger
                    adapt_gens = run_result[5]

                    ind = params['NUM_INDIVS']*run_num
                    fitness_array[ind:ind+params['NUM_INDIVS'], 0:3] = [indiv.fitness.values+(run_num+1,) for indiv in pop]

                    hv_array[:, run_num] = hv

                    num_clusts, aris = evaluation.finalPopMetrics(
                        pop, kwargs['mst_genotype'], final_interest_inds, final_gen_len)
                    num_clusts_array[:, run_num] = num_clusts
                    ari_array[:, run_num] = aris

                    delta_triggers.append(adapt_gens)

                print(f"{strat_name} complete!")

                if validate:
                    print("---------------------------")
                    print("Validating results...")
                    valid = validateResults(
                        os.path.join(os.getcwd(), "test_data", ""),
                        strat_name,
                        ari_array,
                        hv_array,
                        fitness_array,
                        delta_triggers,
                        params['NUM_RUNS']
                        )

                    if not valid:
                        raise ValueError(f"Results incorrect for {strat_name}")

                    else:
                        print(f"{strat_name} validated!\n")


if __name__ == '__main__':
    run_mock(validate=True)