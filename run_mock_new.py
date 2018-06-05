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
import classes
import precompute
import evaluation
import initialisation
import objectives
import delta_mock_mp
from tests import validateResults

def load_data(use_real_data=False, synth_data_subset="*", real_data_subset="*"):
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
    if "synthetic" in file_path:
        classes.Dataset.data_name = file_path.split("/")[-1].split(".")[0][:-15]
    elif "UKC" in file_path:
        classes.Dataset.data_name = file_path.split("/")[-1].split(".")[0]

    with open(file_path) as file:
        head = [int(next(file)[:-1]) for _ in range(4)]

    # Read the data in as an array
    # The skip_header is for the header info in this data specifically
    data = np.genfromtxt(file_path, delimiter="\t", skip_header=4)

    # Set the values for the data
    classes.Dataset.num_examples = head[0] # Num examples
    classes.Dataset.num_features = head[1] # Num features/dimensions
    classes.Dataset.k_user = head[3] # Num real clusters

    # Do we have labels?
    if head[2] == 1:
        classes.Dataset.labels = True
    else:
        classes.Dataset.labels = False

    # Remove labels if present and create data_dict
    data, data_dict = classes.Dataset.createDatasetGarza(data)    

    # Go through the precomputation specific to the dataset
    distarray = precompute.compDists(data, data)
    distarray = precompute.normaliseDistArray(distarray)
    argsortdists = np.argsort(distarray, kind='mergesort')
    nn_rankings = precompute.nnRankings(distarray, classes.Dataset.num_examples)
    classes.MOCKGenotype.mst_genotype = precompute.createMST(distarray)
    classes.MOCKGenotype.degree_int = precompute.degreeInterest(
        classes.MOCKGenotype.mst_genotype, L, nn_rankings, distarray)

    # Finds the indices of the most to least interesting links
    classes.MOCKGenotype.interest_links_indices()

    # Bundle all of the arguments together in a dict to pass to the function
    # This is in order of runMOCK so that we can easily turn it into a partial func for multiprocessing
    kwargs = {
        "data": data,
        "data_dict": data_dict,
        "delta_val": None,
        "hv_ref": None,
        "argsortdists": argsortdists,
        "nn_rankings": nn_rankings,
        # "mst_genotype": mst_genotype,
        # "int_links_indices": int_links_indices,
        "L": L,
        "num_indivs": num_indivs,
        "num_gens": num_gens,
        "delta_reduce": delta_reduce,
        "strat_name": None,
        "adapt_delta": None,
        "relev_links_len": None,
        "reduced_clust_nums": None
        # "seed_num": None
    }

    return kwargs


def create_seeds(NUM_RUNS, seed_file=None, save_new_seeds=True):
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
    try:
        with open(config_path) as json_file:
            params = json.load(json_file)
    except JSONDecodeError:
        print("Unable to load config file")
        raise
    return params


def delta_precomp(data, data_dict, argsortdists, L, delta_val, mst_genotype, int_links_indices):
    """
    Do the precomputation specific to the delta value for that dataset
    """

    relev_links_len = initialisation.relevantLinks(
        delta_val, classes.Dataset.num_examples)
    print("Genotype length:", relev_links_len)

    _, base_clusters = initialisation.baseGenotype(
        mst_genotype, int_links_indices, relev_links_len)

    print(base_clusters, "base_clusters")

    classes.partialClustering(
        base_clusters, data, data_dict, argsortdists, L)

    # Maybe also put this as a class attribute for PartialClust?
    reduced_clust_nums = [
    data_dict[i].base_cluster_num for i in int_links_indices[:relev_links_len]
    ]

    return relev_links_len, reduced_clust_nums

def calc_hv_ref(kwargs, sr_vals):
    """
    Calculates a correct hv reference point
    """

    min_delta = 100-(
        (100*sr_vals[0]*np.sqrt(classes.Dataset.num_examples))/classes.Dataset.num_examples
    )

    relev_links_len, reduced_clust_nums = delta_precomp(
        kwargs['data'], kwargs["data_dict"], kwargs["argsortdists"],
        kwargs["L"], min_delta, kwargs["mst_genotype"], 
        kwargs["int_links_indices"]
    )

    mst_reduced_genotype = [kwargs['mst_genotype'][i] for i in kwargs['int_links_indices'][:relev_links_len]]

    chains, superclusts = objectives.clusterChains(
        mst_reduced_genotype, kwargs['data_dict'], classes.PartialClust.part_clust, reduced_clust_nums
    )
    # print(superclusts)
    # print(classes.PartialClust.base_members)
    
    classes.PartialClust.max_var = objectives.objVAR(
        chains, classes.PartialClust.part_clust, classes.PartialClust.base_members,
        classes.PartialClust.base_centres, superclusts
    )

    hv_ref = [
        classes.PartialClust.max_var*1.05,
        classes.PartialClust.max_conn*1.05
    ]

    print(classes.PartialClust.max_var)

    classes.PartialClust.id_value = count()

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
    # is the same for all runs
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
    print(f"{len(params['strategies'])} strategies")
    print(f"{len(sr_vals)} delta values")
    print(f"{len(data_file_paths)} datasets")
    print(f"= {params['NUM_RUNS']*len(params['strategies'])*len(sr_vals)*len(data_file_paths)} runs")
    print("---------------------------")

    # Loop through the data to test
    for file_path in data_file_paths:
        print(f"Beginning precomputation for {file_path.split(os.sep)[-1]}")
        kwargs = prepare_data(file_path, params['L'], params['NUM_INDIVS'], params['NUM_GENS'])
        print("Precomputation complete")

        if validate:
            kwargs['hv_ref'] = [3.0, 1469.0]

        calc_hv_ref(kwargs, sr_vals)

        # Loop through the sr (square root) values
        for sr_val in sr_vals:
            kwargs['delta_val'] = 100-(
                (100*sr_val*np.sqrt(classes.Dataset.num_examples))/classes.Dataset.num_examples
                )

            if kwargs['delta_val'] > 100:
                raise ValueError("Delta value is too high (over 100)")
            elif kwargs['delta_val'] < 0:
                print("Delta value is below 0, setting to 0...")
                kwargs['delta_val'] = 0
            
            print(f"Delta: {kwargs['delta_val']}")
            classes.PartialClust.id_value = count()
            relev_links_len, reduced_clust_nums = delta_precomp(
                kwargs['data'], kwargs["data_dict"], kwargs["argsortdists"], kwargs["L"], kwargs['delta_val'], 
                kwargs["mst_genotype"], kwargs["int_links_indices"]
                )
            kwargs["relev_links_len"] = relev_links_len
            kwargs["reduced_clust_nums"] = reduced_clust_nums

            # Need to calculate hv reference point here, by calculating VAR for the MST
            # This can be done outside of the sr_val loop
            # May be worth just doing a one-off calc of the min delta value
            # And then calculating the HV_ref from there, which is then fixed for that dataset

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
                start_time = time.time()
                with multiprocessing.Pool() as pool:
                    results = pool.starmap(mock_func, seed_list)
                end_time = time.time()
                print(f"{strat_name} done (took {end_time-start_time:.3f}) - collecting results...")

                for run_num, run_result in enumerate(results):
                    pop = run_result[0]
                    # print([ind.fitness.values for ind in pop])
                    temp_res = sorted([ind.fitness.values for ind in pop], key=lambda var:var[1])
                    print(temp_res[:2])
                    print(temp_res[-2:])

                    hv = run_result[1]
                    # deal with hv_ref
                    int_links_indices_spec = run_result[3]

                    final_relev_links_len = run_result[4]

                    adapt_gens = run_result[5]

                    ind = params['NUM_INDIVS']*run_num
                    fitness_array[ind:ind+params['NUM_INDIVS'], 0:3] = [indiv.fitness.values+(run_num+1,) for indiv in pop]

                    hv_array[:, run_num] = hv

                    num_clusts, aris = evaluation.finalPopMetrics(
                        pop, kwargs['mst_genotype'], int_links_indices_spec, final_relev_links_len)
                    num_clusts_array[:, run_num] = num_clusts
                    ari_array[:, run_num] = aris

                    delta_triggers.append(adapt_gens)

                print(f"{strat_name} complete!")

                if validate:
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
    run_mock()