# Standard libraries
import os
import glob
import json
import random
import time
import numpy as np

# Own functions
import classes
import precompute
import evaluation
import delta_mock
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
    mst_genotype = precompute.createMST(distarray)
    degree_int = precompute.degreeInterest(mst_genotype, L, nn_rankings, distarray)
    int_links_indices = precompute.interestLinksIndices(degree_int)
    print("Precomputation done!\n")

    # Bundle all of the arguments together in a dict to pass to the function
    kwargs = {
        "data": data,
        "data_dict": data_dict,
        "delta_val": None,
        "hv_ref": None,
        "argsortdists": argsortdists,
        "nn_rankings": nn_rankings,
        "mst_genotype": mst_genotype,
        "int_links_indices": int_links_indices,
        "L": L,
        "num_indivs": num_indivs,
        "num_gens": num_gens,
        "delta_reduce": delta_reduce,
        "strat_name": None,
        "adapt_delta": None,
    }

    return kwargs, mst_genotype


def create_seeds(NUM_RUNS, seed_file=None):
    if seed_file is not None:
        params = load_config(seed_file)
        seed_list = params['seed_list']
    else:
        # Randomly generate seeds
        seed_list = [random.uniform(0, 1000) for i in range(NUM_RUNS)]

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


# Maybe this should be main
def run_mock(validate=True):
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

    # Loop through the data to test
    for file_path in data_file_paths:
        print(f"Beginning precomputation for {file_path.split(os.sep)[-1]}")
        kwargs, mst_genotype = prepare_data(file_path, params['L'], params['NUM_INDIVS'], params['NUM_GENS'])
        print("Precomputation complete")

        if validate:
            kwargs['hv_ref'] = [3.0, 1469.0]

        # Loop through the sr values to test
        for sr_val in sr_vals:
            kwargs['delta_val'] = 100-(
                (100*sr_val*np.sqrt(classes.Dataset.num_examples))/classes.Dataset.num_examples
                )

            # Loop through the strategies to test
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

                for run in range(params['NUM_RUNS']):
                    random.seed(seed_list[run])

                    start_time = time.time()
                    pop, hv, hv_ref, int_links_indices_spec, relev_links_len, adapt_gens = delta_mock.runMOCK(**kwargs)
                    end_time = time.time()

                    if run == 0 and hv_ref is not None:
                        print(f"Here at run {run} with ref {hv_ref}")
                        kwargs['hv_ref'] = hv_ref

                    # Add fitness values
                    ind = params['NUM_INDIVS']*run
                    fitness_array[ind:ind+params['NUM_INDIVS'], 0:3] =[indiv.fitness.values+(run+1,) for indiv in pop]

                    # Evaluate the ARI
                    num_clusts, aris = evaluation.finalPopMetrics(
                        pop, mst_genotype, int_links_indices_spec, relev_links_len)
                    
                    num_clusts_array[:, run] = num_clusts
                    ari_array[:, run] = aris
                    hv_array[:, run] = hv
                    time_array[run] = end_time - start_time
                    delta_triggers.append(adapt_gens)
                
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