# Standard libraries
import json
import random
import time
from datetime import datetime
from functools import partial
from pathlib import Path
import multiprocessing

import numpy as np
import pandas as pd

# Own functions
from classes import Datapoint, MOCKGenotype, PartialClust
import precompute
import evaluation
import objectives
from custom_warnings import warning_min_max_delta
import delta_mock
import utils
import tests


def load_data(exp_name, data_folder, validate, data_subset=""):
    # Generate experiment name if not given
    if exp_name is None:
        exp_name = "experiment_" + str(datetime.today().strftime('%Y%m%d'))
    # Create the folder to store the results
    experiment_folder = Path.cwd() / "experiments" / exp_name
    # Warn if already made
    if experiment_folder.is_dir():
        print(f"{experiment_folder} already exists, results may be overwritten")
    if validate is False:
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


def prepare_data(c, X, y):
    data = np.asarray(X)
    target = np.asarray(y)

    # Set the values for the data
    Datapoint.num_examples = data.shape[0]
    Datapoint.num_features = data.shape[1]
    Datapoint.k_user = c.k_user  # Number of presumed real clusters

    if c.verbose:
        print("Num examples:", Datapoint.num_examples)
        print("Num features:", Datapoint.num_features)
        print("Num of clusters (user):", Datapoint.k_user)

    # Remove labels if present and create data_dict
    data, data_dict = Datapoint.create_dataset_garza(data, target)
    return data, data_dict


def setup_mock(data):
    # Go through the precomputation specific to the dataset
    # Calculate distance array
    distarray = precompute.compute_dists(data, data)
    distarray = precompute.normalize_dists(distarray)
    argsortdists = np.argsort(distarray, kind='mergesort')

    # Calculate nearest neighbour rankings
    nn_rankings = precompute.nn_rankings(distarray, Datapoint.num_examples)

    # Calculate MST
    MOCKGenotype.mst_genotype = precompute.create_mst(distarray)
    MOCKGenotype.n_links = len(MOCKGenotype.mst_genotype)

    # Calculate DI values
    MOCKGenotype.degree_int = precompute.degree_interest(
        MOCKGenotype.mst_genotype, nn_rankings, distarray
    )

    # Sort to get the indices of most to least interesting links
    MOCKGenotype.interest_links_indices()

    return argsortdists, nn_rankings


def prepare_mock_args(data_dict, argsortdists, nn_rankings, c):
    init_sr = None
    min_sr = None
    deltas = []
    # Parse 'sr' delta values
    for var in [c.init_delta, c.min_delta, c.max_delta]:
        if isinstance(var, str):
            value = int(var[2:])
            deltas.append(round(MOCKGenotype.calc_delta(value), c.delta_precision))
        else:
            deltas.append(var)

    if c.domain_delta == 'sr':
        min_sr = int(c.min_delta[2:])
        init_sr = int(c.init_delta[2:])
        init_sr, min_sr = warning_min_max_delta(init_sr, min_sr)

    deltas[1], deltas[0] = warning_min_max_delta(deltas[1], deltas[0])

    # Bundle all of the arguments together in a dict to pass to the function
    # This is in order of runMOCK() so that we can easily turn it into a partial func for multiprocessing
    mock_args = {
        # "data": data,
        "data_dict": data_dict,
        "hv_ref": None,
        "argsortdists": argsortdists,
        "nn_rankings": nn_rankings,
        "L": c.L,
        "num_indvs": c.num_indvs,
        "num_gens": c.num_gens,
        "mut_meth_params": None,
        "domain": c.domain_delta,
        "init_delta": deltas[0],
        "min_delta": deltas[1],
        "max_delta": deltas[2],
        "init_sr": init_sr,
        "min_sr": min_sr,
        "flexible_limits": c.flexible_limits,
        "stair_limits": c.stair_limits,
        "gens_step": c.gens_step,
        "delta_mutation": c.delta_mutation,
        "squash": c.squash,
        "delta_precision": c.delta_precision,
        "delta_mutpb": c.delta_mutation_probability,
        "delta_sigma": c.delta_mutation_variance,
        "delta_sigma_as_perct": c.delta_as_perct,
        "delta_inverse": c.delta_inverse,
        "crossover": c.crossover,
        "save_history": c.save_history,
        "verbose": c.verbose
    }

    return mock_args


def create_seeds(config):
    # Specify seed folder
    seed_folder = Path.cwd() / "seeds"
    # Load seeds if present
    if config["seed_file"] is not None:
        seed_list = utils.load_json(seed_folder / config["seed_file"])["seed_list"]
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
        seed_fname = seed_folder / f"seed_list_{config['exp_name']}.json"
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
    PartialClust.max_var = PartialClust.max_var / Datapoint.num_examples
    # Set reference point just outside max values to ensure no overlap
    hv_ref = [
        PartialClust.max_var * 1.01,
        PartialClust.max_cnn * 1.01
    ]
    return hv_ref


def single_run_mock(seed, run_number, mock_args, data, c, validate):
    # Setup some of the variables for the genotype
    MOCKGenotype.setup_genotype_vars(min_delta=mock_args['init_delta'],
                                     data=data,
                                     data_dict=mock_args["data_dict"],
                                     argsortdists=mock_args["argsortdists"],
                                     L=mock_args["L"],
                                     domain=mock_args['domain'],
                                     max_sr=mock_args['init_sr'])

    if mock_args['min_sr'] is not None and mock_args['min_sr'] > MOCKGenotype.sr_upper_bound:
        mock_args['min_sr'] = MOCKGenotype.sr_upper_bound

    # Set the nadir point
    if validate:
        # To ensure compatible results for validation
        mock_args['hv_ref'] = [3.0, 1469.0]
    else:
        mock_args['hv_ref'] = calc_hv_ref(mock_args)

    # Add mutation method-specific arguments
    mock_args = delta_mock.get_mutation_params(c.mut_method, mock_args, c.L_comp)

    # Run MOCK
    start_time = time.time()
    result = delta_mock.runMOCK(seed_num=seed, data=data, run_number=run_number, **mock_args)
    mp_time = time.time() - start_time

    if c.verbose:
        print(f"Run {run_number} completed... It took {mp_time:.3f} secs)")

    # Extract the population
    all_pop = result[0]
    # Extract hv list
    hvs = result[1]
    # Extract final interesting links. For future ref.
    final_interest_inds = result[3]
    # Extract final genotype length. For future ref.
    final_gen_len = result[4]
    # Get the running time for each run
    time_taken = result[5]
    # Calculate the number of clusters and ARIs
    clust_aris = [evaluation.final_pop_metrics(pop) for pop in all_pop]
    aris = []
    num_clusts = []
    for clust_ari in clust_aris:
        num_clusts.append(clust_ari[0])
        aris.append(clust_ari[1])

    # Get the labels [2] for the best solution [np.argmax(aris[-1])] in the last generation [-1]
    pred_labels = clust_aris[-1][2][np.argmax(aris[-1])]

    # Extract the fitness values
    var_vals = [[indiv.fitness.values[0] for indiv in pop] for pop in all_pop]
    cnn_vals = [[indiv.fitness.values[1] for indiv in pop] for pop in all_pop]
    delta_vals = [[indiv.delta for indiv in pop] for pop in all_pop]
    # Flatten list of lists
    var_vals = np.array(var_vals).flatten()
    cnn_vals = np.array(cnn_vals).flatten()
    delta_vals = np.array(delta_vals).flatten()
    aris = np.array(aris).flatten() if aris[0] is not None else np.full_like(delta_vals, np.nan)
    num_clusts = np.array(num_clusts).flatten()

    # Generation numbers
    gen_number = []
    if c.save_history:
        for i in range(1, len(all_pop)+1):
            gen_number += [i] * c.num_indvs
    else:
        gen_number += [c.num_gens] * c.num_indvs

    # Add strategy here for adaptive version
    n_obs = c.num_indvs * len(all_pop)
    results_dict = {
        "dataset": [Datapoint.data_name] * n_obs,
        'run': [run_number] * n_obs,
        'gen': gen_number,
        "indiv": list(range(c.num_indvs)) * len(all_pop),
        "L": [c.L] * n_obs,
        "delta": delta_vals,
        "min_delta": [mock_args['min_delta']] * n_obs,
        "init_delta": [mock_args['init_delta']] * n_obs,
        "max_delta": [mock_args['max_delta']] * n_obs,
        "delta_mutation_probability": [c.delta_mutation_probability] * n_obs,
        "delta_mutation_variance": [c.delta_mutation_variance] * n_obs,
        "delta_as_perct": [c.delta_as_perct] * n_obs,
        "delta_inverse": [c.delta_inverse] * n_obs,
        "VAR": var_vals,
        "CNN": cnn_vals,
        "ARI": aris,
        "clusters": num_clusts,
        "time": [time_taken] * n_obs
    }
    # Add the new results
    results_df = pd.DataFrame(results_dict)
    hvs_df = pd.DataFrame({'indiv': [i for i in range(len(hvs))],
                           'run': [run_number] * len(hvs),
                           'HV': hvs})

    return results_df, hvs_df, pred_labels


def multi_run_mock(c, X, y):
    """Main flow controller for MOCK
    :param c: PyMOCK instance
    :param X: Features.
    :param y: Labels (if available)"""
    if c.verbose:
        # Print the number of runs to get an idea of runtime (sort of)
        print("--*"*15)
        print("Init information:")
        print(f"{c.num_runs} run(s)")
        print(f"{c.num_gens} generations")
        print(f"{c.num_indvs} individuals per generation.")
        print(f"Init delta: {c.init_delta}.")
        print(f"Min delta: {c.min_delta}.")
        print("--*"*15)

        print(f"Beginning precomputation...")

    # Prepare the data
    data, data_dict = prepare_data(c, X, y)
    # Do the precomputation for MOCK
    argsortdists, nn_rankings = setup_mock(data)
    # Wrap the arguments up for the main MOCK function
    mock_args = prepare_mock_args(data_dict, argsortdists, nn_rankings, c)

    if c.verbose:
        print(f'Min delta: {mock_args["min_delta"]}')
        print(f'Init delta: {mock_args["init_delta"]}')
        print("Precomputation complete!")
        print("--*"*15)
        print('Beginning {} runs.'.format(c.num_runs))

    if c.num_runs == 1:
        # Don't enter multiprocessing if only one run
        results = [single_run_mock(mock_args=mock_args, validate=c.validate,
                                   data=data, c=c, **c.runs_list[0])]
    else:
        mock_func = partial(single_run_mock, mock_args=mock_args, validate=c.validate,
                            data=data, c=c)
        with multiprocessing.Pool() as pool:
            results = pool.starmap(mock_func, c.runs_list)

    # Save results back into instance
    c.results_df = pd.DataFrame()
    c.hvs_df = pd.DataFrame()
    c.labels = []
    for result in results:
        c.results_df = c.results_df.append(result[0], sort=False)
        c.hvs_df = c.hvs_df.append(result[1], sort=False)
        c.labels.append(result[2])

    if c.verbose:
        print(f"Done!")

    # Validate the results
    if c.validate:
        tests.validate_results(c.results_df)
        print("Test passed!")
