 
# Standard libraries
import os
import glob
import json
import random
import time
import csv
from functools import partial
import multiprocessing
import numpy as np

# Own functions
from classes import Dataset, MOCKGenotype, PartialClust
import precompute
import evaluation
import objectives
import delta_mock
import utils
from tests import validate_results

def load_data(use_real_data=False, synth_data_subset="*", real_data_subset="*", exp_name=""):
    """Get the file paths for all the data we're using
    
    Keyword Arguments:
        use_real_data {bool} -- The real data consumes a lot of memory, can choose to exclude it (default: {False})
        synth_data_subset {str} -- Allows for selecting specific datasets from the synthetic set (default: {"*"})
        real_data_subset {str} -- As above for the real (UKC) data (default: {"*"})
        exp_name {str} -- Name for particular experiment to create subfolder in results directory
    
    Returns:
        [list] -- List of all the data files
        results_folder [str] -- Where the results should be saved
    """
    # Get the current directory path
    base_path = os.getcwd()

    data_folder = os.path.join(base_path, "data", "")

    if exp_name != "":
        results_path = os.path.join(base_path, "results", exp_name)
        try:
            os.makedirs(results_path)
            results_folder = os.path.join(results_path, "")
        except FileExistsError:
            results_folder = os.path.join(results_path, "")
    else:
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
    
    print("Num examples:", Dataset.num_examples)
    print("Num features:", Dataset.num_features)
    print("Num (actual) clusters:", Dataset.k_user)

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
    MOCKGenotype.degree_int = precompute.degreeInterest(MOCKGenotype.mst_genotype, nn_rankings, distarray)
    
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
        "reduced_clust_nums": None,
        "mut_meth_params": None
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
        params = utils.load_json(seed_file)
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


def calc_hv_ref(kwargs):
    """Calculates a hv reference/nadir point for use on all runs of that dataset
    
    Arguments:
        kwargs {dict} -- Arguments for MOCK
    
    Returns:
        [list] -- The reference/nadir point
    """

    mst_reduced_genotype = [MOCKGenotype.mst_genotype[i] for i in MOCKGenotype.reduced_genotype_indices]

    # Calculate chains
    chains, superclusts = objectives.clusterChains(
        mst_reduced_genotype, kwargs['data_dict'], PartialClust.comp_dict, MOCKGenotype.reduced_cluster_nums
    )
    
    PartialClust.max_var = objectives.objVAR(
        chains, PartialClust.comp_dict, PartialClust.base_members,
        PartialClust.base_centres, superclusts
    )

    # Need to divide by N as this is done in evalMOCK() not objVAR()
    PartialClust.max_var= PartialClust.max_var/Dataset.num_examples

    # Set reference point just outside max values to ensure no overlap
    hv_ref = [
        PartialClust.max_var*1.01,
        PartialClust.max_cnn*1.01
    ]
    return hv_ref

def run_mock(**cl_args):
    # Load the data file paths
    if cl_args['validate']:
        data_file_paths, results_folder = load_data(
            synth_data_subset="tevc_20_60_9*")
    else:
        data_file_paths, results_folder = load_data(
            synth_data_subset=cl_args['synthdata'],
            exp_name=cl_args['exp_name'])

    # Load general MOCK parameyers
    params = utils.load_json("mock_config.json")

    # A longer genotype means a higher possible maximum for the CNN objective
    # By running the highest sr value first, we ensure that the HV_ref
    # is the same and appropriate for all runs
    sr_vals = sorted(params['sr_vals'], reverse=True)

    # Column names for fitness array, formatted for EAF R plot package
    fitness_cols = ["VAR", "CNN", "Run"]

    #####
    # move num runs to a cl arg
    # move num indivs to a cl arg
    # move L to a cl_arg
    # move num gens to a cl arg

    # try:
    #     cl_args['num_runs'] = cl_args['num_runs']
    #     cl_args['num_runs'] = cl_args['num_runs']
    #     cl_args['num_runs'] = cl_args['num_runs']
    # except KeyError as e:
    #     print(f"Argument not found: {e}")

    # Create the seed list or load existing one
    if cl_args['validate']:
        sr_vals = [1]
        cl_args['num_runs'] = 4
        
        seed_list = create_seeds(
            cl_args['num_runs'], seed_file="seed_list_validate.json")

    else:
        seed_list = create_seeds(cl_args['num_runs'])

    # Restrict seed_list to the actual number of runs that we need
    # Truncating like this allows us to know that the run numbers and order of seeds correspond
    seed_list = seed_list[:cl_args['num_runs']]
    seed_list = [(i,) for i in seed_list]

    # Print the number of runs to get an idea of runtime (sort of)
    print("---------------------------")
    print("Number of MOCK Runs:")
    print(f"{cl_args['num_runs']} run(s)")
    print(f"{len(params['strategies'])} strategy/-ies")
    print(f"{len(sr_vals)} delta value(s)")
    print(f"{len(data_file_paths)} dataset(s)")
    print(f"= {cl_args['num_runs']*len(params['strategies'])*len(sr_vals)*len(data_file_paths)} run(s)")
    print("---------------------------")

    # Loop through the data to test
    for file_path in data_file_paths:
        print(f"Beginning precomputation for {file_path.split(os.sep)[-1]}...")
        
        kwargs = prepare_data(file_path, cl_args['L'], cl_args['num_indivs'], cl_args['num_gens'])
        print("Precomputation complete!")

        # Loop through the sr (square root) values
        for sr_val in sr_vals:
            # Calculate the delta value from the sr
            MOCKGenotype.calc_delta(sr_val)
            kwargs['delta_val'] = MOCKGenotype.delta_val
            
            print(f"Delta: {kwargs['delta_val']}")

            # Setup some of the variables for the genotype
            MOCKGenotype.setup_genotype_vars()

            PartialClust.partial_clusts(kwargs["data"], kwargs["data_dict"], kwargs["argsortdists"], kwargs["L"])
            MOCKGenotype.calc_reduced_clusts(kwargs["data_dict"])

            # Set the nadir point if first run
            if kwargs['hv_ref'] is None:
                # To ensure compatible results
                if cl_args['validate']:
                    kwargs['hv_ref'] = [3.0, 1469.0]
                else:
                    kwargs['hv_ref'] = calc_hv_ref(kwargs)

            if cl_args['mut_method'] == "centroid":
                distarray_cen = precompute.compDists(
                    PartialClust.base_centres, PartialClust.base_centres)
                kwargs['mut_meth_params'] = {
                    'mut_method': "centroid",
                    'argsortdists_cen': np.argsort(
                        distarray_cen, kind='mergesort'),
                    'nn_rankings_cen': precompute.nnRankings(
                        distarray_cen, len(PartialClust.comp_dict))
                }                
            elif cl_args['mut_method'] == "neighbour":
                kwargs['mut_meth_params'] = {
                    'mut_method': "neighbour",
                    'component_nns': precompute.nn_comps(
                        Dataset.num_examples, kwargs['argsortdists'], kwargs['data_dict'], kwargs['L'])
                }
            else:
                kwargs['mut_meth_params'] = {
                    'mut_method': "original"
                }
            # print(kwargs['mut_meth_params'])
            # calc_hv_ref(kwargs)
            print(f"HV ref point: {kwargs['hv_ref']}")
            
            # Loop through the strategies
            for strat_name in params['strategies']:
                kwargs['strat_name'] = strat_name
                if strat_name == "base":
                    kwargs['adapt_delta'] = False
                else:
                    kwargs['adapt_delta'] = True

                fitness_array = np.empty((cl_args['num_indivs']*cl_args['num_runs'], len(fitness_cols)))
                hv_array = np.empty((cl_args['num_gens'], cl_args['num_runs']))
                ari_array = np.empty((cl_args['num_indivs'], cl_args['num_runs']))
                num_clusts_array = np.empty((cl_args['num_indivs'], cl_args['num_runs']))
                time_array = np.empty(cl_args['num_runs'])
                delta_triggers = []

                # Abstract the below to a precompute func
                # Can then choose which based on the mutation method
                                

                # kwargs['argsortdists_cen'] = np.argsort(distarray_cen, kind='mergesort')
                # kwargs['nn_rankings_cen'] = precompute.nnRankings_cen(distarray_cen, len(PartialClust.comp_dict))
                
                mock_func = partial(delta_mock.runMOCK, *list(kwargs.values()))

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
                    # Extract hv list
                    hv = run_result[1]
                    # Extract final interesting links
                    final_interest_inds = run_result[3]
                    # Extract final genotype length
                    final_gen_len = run_result[4]
                    # Extract gens with delta trigger
                    adapt_gens = run_result[5]

                    # Assign values to arrays
                    ind = cl_args['num_indivs']*run_num
                    fitness_array[ind:ind+cl_args['num_indivs'], 0:3] = [indiv.fitness.values+(run_num+1,) for indiv in pop]

                    hv_array[:, run_num] = hv

                    num_clusts, aris = evaluation.finalPopMetrics(
                        pop, kwargs['mst_genotype'], final_interest_inds, final_gen_len)
                    num_clusts_array[:, run_num] = num_clusts
                    ari_array[:, run_num] = aris

                    delta_triggers.append(adapt_gens)

                print(f"{strat_name} complete!")

                if cl_args['validate']:
                    print("---------------------------")
                    print("Validating results...")
                    valid = validate_results(
                        os.path.join(os.getcwd(), "test_data", ""),
                        strat_name,
                        ari_array,
                        hv_array,
                        fitness_array,
                        delta_triggers,
                        cl_args['num_runs']
                        )

                    if not valid:
                        raise ValueError(f"Results incorrect for {strat_name}")

                    else:
                        print(f"{strat_name} validated!\n")

                if cl_args['exp_name'] != "":
                    fname_prefix = "-".join(
                        [results_folder+Dataset.data_name, strat_name])
                    print(fname_prefix)
                    
                    if kwargs['adapt_delta']:
                        fname_suffix = "-adapt"
                    else:
                        fname_suffix = ""

                    # Save fitness values
                    np.savetxt(
                        fname_prefix+"-fitness-sr"+str(sr_val)+fname_suffix+".csv", fitness_array, 
                        delimiter=",")
                    # Save hypervolume values
                    np.savetxt(
                        fname_prefix+"-hv-sr"+str(sr_val)+fname_suffix+".csv", hv_array, 
                        delimiter=",")
                    # Save ARI values
                    np.savetxt(
                        fname_prefix+"-ari-sr"+str(sr_val)+fname_suffix+".csv", ari_array, 
                        delimiter=",")
                    # Save number of clusters
                    np.savetxt(
                        fname_prefix+"-numclusts-sr"+str(sr_val)+fname_suffix+".csv", num_clusts_array, 
                        delimiter=",")
                    # Save computation time
                    np.savetxt(
                        fname_prefix+"-time-sr"+str(sr_val)+fname_suffix+".csv", time_array, 
                        delimiter=",")

                    # Save delta triggers
                    # No triggers for normal delta-MOCK
                    if kwargs['adapt_delta']:
                        with open(
                            fname_prefix+"-triggers-sr"+str(sr_val)+fname_suffix+".csv","w") as f:
                            writer=csv.writer(f)
                            writer.writerows(delta_triggers)

if __name__ == '__main__':
    parser = utils.build_parser()
    cl_args = parser.parse_args()
    cl_args = vars(cl_args)
    utils.check_cl_args(cl_args)

    run_mock(**cl_args)

    ######## TO DO ########
    # add hook for crossover
    # try to clean up arguments generally
    # look at how results are saved and named
    # then look at generating graphs
    # send to servers and run