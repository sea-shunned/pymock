# Standard libraries
import os
import glob
import json
import random
import time
import csv
from itertools import product
from datetime import datetime
from functools import partial
from pathlib import Path
import multiprocessing
import pdb

import numpy as np
import pandas as pd

# Own functions
from classes import Datapoint, MOCKGenotype, PartialClust
import precompute
import evaluation
import objectives
import delta_mock
import utils
import tests

def load_data(exp_name, data_folder, data_subset=""):
    # Generate experiment name if not given
    if exp_name is None:
        exp_name = f"experiment_{datetime.today().strftime('%Y%m%d')}"
    # Create the folder to store the results
    experiment_folder = Path.cwd() / "experiments" / exp_name
    # Warn if already made
    if experiment_folder.is_dir():
        print(f"{experiment_folder} already exists, results may be overwritten")
    # Make the folder if not already
    experiment_folder.mkdir(parents=True, exist_ok=True)
    # Turn the folder into a Path
    # If a relative path to the data is given this splits it properly for cross-platform
    data_folder = Path.cwd().joinpath(*[i for i in data_folder.split("/")])
    # Check if the data_folder exists
    if not data_folder.is_dir():
        raise NotADirectoryError(f"{data_folder} cannot be found")
    # Select the datasets from the folder if a filter is given
    if data_subset is None:
        data_file_paths = data_folder.glob("*")
    else:
        data_file_paths = data_folder.glob("*"+data_subset+"*")
    # Return the data and the base experiment folder
    return list(data_file_paths), experiment_folder

def prepare_data(file_path):
    # Some of this function is hard-coded for strings, so convert the Path
    if isinstance(file_path, Path):
        file_path = str(file_path)
    # Get prettier names for the data
    if "synthetic" in file_path:
        Datapoint.data_name = file_path.split("/")[-1].split(".")[0][:-15]
    elif "UKC" in file_path:
        Datapoint.data_name = file_path.split("/")[-1].split(".")[0]
    # Current data has header with metadata
    with open(file_path) as file:
        head = [int(next(file)[:-1]) for _ in range(4)]
    # Read the data in as an array
    # The skip_header is for the header info in this data specifically
    data = np.genfromtxt(file_path, delimiter="\t", skip_header=4)

    # Set the values for the data
    Datapoint.num_examples = head[0] # Num examples
    Datapoint.num_features = head[1] # Num features/dimensions
    Datapoint.k_user = head[3] # Num real clusters
    
    print("Num examples:", Datapoint.num_examples)
    print("Num features:", Datapoint.num_features)
    print("Num (actual) clusters:", Datapoint.k_user)
    # Do we have labels?
    if head[2] == 1:
        Datapoint.labels = True
    else:
        Datapoint.labels = False

    # Remove labels if present and create data_dict
    data, data_dict = Datapoint.create_dataset_garza(data)
    return data, data_dict

def setup_mock(data, data_dict):
    # Go through the precomputation specific to the dataset
    # Calculate distance array
    distarray = precompute.compute_dists(data, data)
    distarray = precompute.normalize_dists(distarray)
    argsortdists = np.argsort(distarray, kind='mergesort')
    # Calculate nearest neighbour rankings
    nn_rankings = precompute.nn_rankings(distarray, Datapoint.num_examples)
    # Calculate MST
    MOCKGenotype.mst_genotype = precompute.create_mst(distarray)
    # Calculate DI values
    MOCKGenotype.degree_int = precompute.degree_interest(
        MOCKGenotype.mst_genotype, nn_rankings, distarray
    )
    # Sort to get the indices of most to least interesting links
    MOCKGenotype.interest_links_indices()
    return argsortdists, nn_rankings

def prepare_mock_args(data, data_dict, argsortdists, nn_rankings, config):
    # Bundle all of the arguments together in a dict to pass to the function
    # This is in order of runMOCK() so that we can easily turn it into a partial func for multiprocessing
    mock_args = {
        "data": data,
        "data_dict": data_dict,
        "hv_ref": None,
        "argsortdists": argsortdists,
        "nn_rankings": nn_rankings,
        "L": None,
        "num_indivs": config["num_indivs"],
        "num_gens": config["num_gens"],
        "strategy": None,
        "adapt_delta": None,
        "mut_meth_params": None
    }
    return mock_args

def create_seeds(config, experiment_folder, validate):
    # Need a special case just because we like to keep the validation folder separate
    if validate:
        seed_list = utils.load_json(Path.cwd() / "validation" / config["seed_file"])["seed_list"]
    else:
        # Load seeds if present
        if config["seed_file"] is not None:
            seed_list = utils.load_json(experiment_folder / config["seed_file"])["seed_list"]
        # Otherwise make some seeds
        else:
            # Ensure no collision of seeds
            while True:
                # Randomly generate seeds
                seed_list = [
                    random.randint(0, 1000000) for i in range(config["num_runs"])
                ]
                # If no collisions, break
                if len(seed_list) == len(set(seed_list)):
                    break
            # Save the seed_list in the results folder
            seed_fname = experiment_folder / f"seed_list_{config['exp_name']}.json"
            # Create a dict to save into the json just to be specific
            seeds = {"seed_list": seed_list}
            # Save the seeds
            with open(seed_fname, "w") as out_file:
                json.dump(seeds, out_file, indent=4)
            # Add the seed list to the config file so when we save it it's complete
            config["seed_file"] = f"seed_list_{config['exp_name']}.json"
    # Ensure we have enough seeds
    if len(seed_list) < config["num_runs"]:
        raise ValueError("Not enough seeds for number of runs")
    return seed_list, config

def calc_hv_ref(mock_args):
    """Calculates a hv reference/nadir point for use on all runs of that dataset
    
    Arguments:
        mock_args {dict} -- Arguments for MOCK
    
    Returns:
        [list] -- The reference/nadir point
    """
    # Reduce the MST genotype
    mst_reduced_genotype = [MOCKGenotype.mst_genotype[i] for i in MOCKGenotype.reduced_genotype_indices]
    # Calculate chains
    chains, superclusts = objectives.cluster_chains(
        mst_reduced_genotype, mock_args['data_dict'], PartialClust.comp_dict, MOCKGenotype.reduced_cluster_nums
    )
    # Calculate the maximum possible intracluster variance
    PartialClust.max_var = objectives.objVAR(
        chains, PartialClust.comp_dict, PartialClust.base_members,
        PartialClust.base_centres, superclusts
    )
    # Need to divide by N as this is done in eval_mock() not objVAR()
    PartialClust.max_var= PartialClust.max_var/Datapoint.num_examples
    # Set reference point just outside max values to ensure no overlap
    hv_ref = [
        PartialClust.max_var*1.01,
        PartialClust.max_cnn*1.01
    ]
    return hv_ref

def run_mock(**cl_args):
    # Load the data file paths
    if cl_args['validate']:
        config_path = Path.cwd() / "configs" / "validate.json"
        config = utils.load_json(config_path)
        data_file_paths, experiment_folder = load_data(
            config["exp_name"],
            config["data_folder"],
            config["data_subset"]
        )
        # Just validating so don't save results
        save_results = True
    else:
        config_path = Path.cwd() / "configs" / cl_args["config"]
        config = utils.load_json(config_path)
        data_file_paths, experiment_folder = load_data(
            config["exp_name"],
            config["data_folder"],
            config["data_subset"]
        )
        # Save experimental results
        save_results = True

    # Check the config and amend if needed
    config = utils.check_config(config)
    # Create the seed list
    seed_list, config = create_seeds(
        config,
        experiment_folder,
        cl_args["validate"]
    )
    # Restrict seed_list to the actual number of runs that we need
    # Truncating like this allows us to know that the run numbers and order of seeds correspond
    seed_list = seed_list[:config["num_runs"]]
    seed_list = [(i,) for i in seed_list] # for starmap

    # Calculate number of delta values
    # Just needed here to print
    if config["delta_sr_vals"] is None:
        a = 0
    else:
        a = len(config["delta_sr_vals"])

    if config["delta_raw_vals"] is None:
        b = 0
    else:
        b = len(config["delta_raw_vals"])
    num_delta = a + b

    # Print the number of runs to get an idea of runtime (sort of)
    print("---------------------------")
    print("Number of MOCK Runs:")
    print(f"{config['num_runs']} run(s)")
    print(f"{len(config['strategies'])} strategy/-ies")
    print(f"{num_delta} delta value(s)")
    print(f"{len(data_file_paths)} dataset(s)")
    print(f"= {config['num_runs']*len(config['strategies'])*num_delta*len(data_file_paths)} run(s)")
    print("---------------------------")

    # Save the config
    utils.save_config(config, experiment_folder)
    # Make a sub-folder to save results in
    if save_results:
        results_folder = experiment_folder / "results"
        results_folder.mkdir(exist_ok=True)

    # Loop through the data to test
    for file_path in data_file_paths:
        print(f"Beginning precomputation for {file_path.name}...")
        # Prepare the data
        data, data_dict = prepare_data(file_path)
        # Do the precomputation for MOCK
        argsortdists, nn_rankings = setup_mock(data, data_dict)
        # Wrap the arguments up for the main MOCK function
        mock_args = prepare_mock_args(data, data_dict, argsortdists, nn_rankings, config)
        print("Precomputation complete!")

        if save_results:
            # Create a dataframe for the results
            results_df = pd.DataFrame()

        # A longer genotype means a higher possible maximum for the CNN objective
        # By running the highest sr value first, we ensure that the HV_ref
        # is the same and appropriate for all runs    
        # If both sr and raw delta values are given, convert and unify them
        if config["delta_sr_vals"] is not None and config["delta_raw_vals"] is not None:
            delta_vals = sorted(
                config["delta_raw_vals"] + [
                    MOCKGenotype.calc_delta(sr_val) for sr_val in config["delta_sr_vals"]
                ],
                reverse=True
            )
        # If just raw values, then just reverse sort them
        elif config["delta_sr_vals"] is None and config["delta_raw_vals"] is not None:
            delta_vals = sorted(config['delta_raw_vals'], reverse=True)
        # If just sr values, then convert and reverse sort them
        elif config["delta_sr_vals"] is not None and config["delta_raw_vals"] is None:
            delta_vals = sorted(
                [MOCKGenotype.calc_delta(sr_val) for sr_val in config["delta_sr_vals"]],
                reverse=True
            )

        # Loop through the sr (square root) values
        for delta_val, L in product(delta_vals, config["L"]):
            # Set the delta value in the args
            # mock_args['delta_val'] = MOCKGenotype.delta_val
            MOCKGenotype.delta_val = delta_val
            print(f"Delta: {MOCKGenotype.delta_val}")

            # Set the mock_args for this layer
            mock_args["L"] = L

            # Setup some of the variables for the genotype
            MOCKGenotype.setup_genotype_vars()
            # Setup the components class
            PartialClust.partial_clusts(
                mock_args["data"], mock_args["data_dict"], mock_args["argsortdists"], mock_args["L"]
            )
            # Identify the component IDs of the link origins
            MOCKGenotype.calc_reduced_clusts(mock_args["data_dict"])

            # Set the nadir point if first run
            if mock_args['hv_ref'] is None:
                # To ensure compatible results for validation
                if cl_args['validate']:
                    mock_args['hv_ref'] = [3.0, 1469.0]
                else:
                    mock_args['hv_ref'] = calc_hv_ref(mock_args)
            print(f"HV ref point: {mock_args['hv_ref']}")
            
            # Avoid more nested loops
            for strategy, L_comp in product(
                    config["strategies"], config["L_comp"]
                ):
                # Set the mock_args for this layer
                mock_args["strategy"] = strategy
                # mock_args["L_comp"] = L_comp
                # Add mutation method-specific arguments
                mock_args = delta_mock.get_mutation_params(
                    config["mut_method"], mock_args, L_comp
                )
                # Add the strat name to the mock_args
                mock_args['strategy'] = strategy
                # Adaptation flag to make it easier to process
                if strategy == "base":
                    mock_args['adapt_delta'] = False
                else:
                    mock_args['adapt_delta'] = True
                # Initialize arrays for the results
                if save_results and mock_args["adapt_delta"]:
                    delta_triggers = [] # which generation delta was changed
                
                # print(mock_args.keys())
                # for key, val in mock_args.items():
                #     if key not in ["data", "data_dict"]:
                #         print(key, val)
                # raise
                
                # print(mock_args.values())
                # Create the partial function to give to multiprocessing
                # mock_func = partial(delta_mock.runMOCK, *list(mock_args.values()))
                mock_func = partial(delta_mock.runMOCK, **mock_args)
                # print(mock_func.args.keys())
                # print(mock_func.keywords)

                print(f"{strategy}-{config['mut_method']} starting...")
                # Measure the time taken for the runs
                start_time = time.time()
                # Send the function to a thread, each thread with a different seed
                with multiprocessing.Pool() as pool:
                    results = pool.starmap(mock_func, seed_list)
                end_time = time.time()
                # This does not given you a time per run
                # Diving by number of seeds is also inaccurate, it depends on chunksize, num CPUs etc.
                # These numbers are useful to compare relatively, though
                time_taken = end_time-start_time
                print(f"{strategy} done ({len(seed_list)} runs took {time_taken:.3f} secs)")

                for run_num, run_result in enumerate(results):
                    # Extract the population
                    pop = run_result[0]
                    # Extract hv list
                    hvs = run_result[1]
                    # Extract final interesting links
                    final_interest_inds = run_result[3]
                    # Extract final genotype length
                    final_gen_len = run_result[4]
                    # Extract gens with delta trigger
                    adapt_gens = run_result[5]

                    ## Shorten the above #22.05.19
                    # pop, hv, final_interest_inds, final_gen_len, adapt_gens = run_result
                    # Calculate the number of clusters and ARIs
                    num_clusts, aris = evaluation.final_pop_metrics(
                        pop, MOCKGenotype.mst_genotype,
                        final_interest_inds, final_gen_len
                    )
                    # Extract the fitness values
                    var_vals = [indiv.fitness.values[0] for indiv in pop]
                    cnn_vals = [indiv.fitness.values[1] for indiv in pop]
                    # Add strategy here for adaptive version
                    results_dict = {
                        "dataset": [Datapoint.data_name]*config["num_indivs"],
                        "run": [run_num+1]*config["num_indivs"],
                        "indiv": list(range(config["num_indivs"])),
                        "L": [L]*config["num_indivs"],
                        "delta": [delta_val]*config["num_indivs"],
                        "VAR": var_vals,
                        "CNN": cnn_vals,
                        "HV": hvs,
                        "ARI": aris,
                        "clusters": num_clusts,
                        "time": [time_taken]*config["num_indivs"]
                    }
                    results_df = results_df.append(
                        pd.DataFrame.from_dict(results_dict), ignore_index=True
                    )
                    
                    # if L_comp is not None:
                    if save_results and mock_args["adapt_delta"]:
                        delta_triggers.append(adapt_gens)

        # print(results_df)
        print(f"{file_path.name} complete!")

    # Validate the results
    if cl_args['validate']:
        valid = tests.validate_mock(
            results_df, delta_triggers=[],
            strategy=strategy, num_runs=config["num_runs"]
        )
        if valid:
            print("Passed!")
        else:
            raise ValueError(f"Results incorrect!")
    # Save results
    if save_results:
        results_df.to_csv(str(results_folder)+f"{config['exp_name']}_results.csv")

if __name__ == '__main__':
    parser = utils.build_parser()
    cl_args = parser.parse_args()
    cl_args = vars(cl_args)
    utils.check_cl_args(cl_args)

    run_mock(**cl_args)

    ######## TO DO ########
    # look at how results are saved and named
    # evaluation.py is a shit show
    # should clean up and define the required environment at some point
