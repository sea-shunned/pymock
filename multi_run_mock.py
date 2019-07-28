# Standard libraries
import json
import random
import time
from itertools import product
from datetime import datetime
from functools import partial
from pathlib import Path
import multiprocessing
# import pdb

import numpy as np
import pandas as pd

# Own functions
from classes import Datapoint, MOCKGenotype, PartialClust
import precompute
import evaluation
import objectives
from custom_warnings import warning_min_max_delta
# import no_precomp_objectives
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


def prepare_data(file_path):
    # Some of this function is hard-coded for strings, so convert the Path
    if isinstance(file_path, Path):
        file_path = str(file_path)
    # Get prettier names for the data
    if "tevc" in file_path:
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


def prepare_mock_args(data_dict, argsortdists, nn_rankings, config):
    # Bundle all of the arguments together in a dict to pass to the function
    # This is in order of runMOCK() so that we can easily turn it into a partial func for multiprocessing
    mock_args = {
        # "data": data,
        "data_dict": data_dict,
        "hv_ref": None,
        "argsortdists": argsortdists,
        "nn_rankings": nn_rankings,
        "L": None,
        "num_indivs": config["num_indivs"],
        "num_gens": config["num_gens"],
        "mut_meth_params": None,
        "init_delta": config["init_delta"],
        "min_delta": config["min_delta"],
        "max_delta": 100 - config['delta_precision'],
        "flexible_limits": config['flexible_limits'],
        "stair_limits": config['stair_limits'],
        "gens_step": config['gens_step'],
        "delta_mutation": config['delta_mutation'],
        "squash": config['squash'],
        "delta_precision": config['delta_precision'],
        "delta_mutpb": None,
        "delta_sigma": None,
        "delta_sigma_as_perct": None,
        "delta_inverse": None,
        "crossover": config['crossover'],
        "save_history": config['save_history'],
        "verbose": config['verbose']
    }

    for var in ['min_delta', 'init_delta']:
        if isinstance(mock_args[var], str):
            value = float(mock_args[var][2:])
            mock_args[var] = round(MOCKGenotype.calc_delta(value), config['delta_precision'])
    mock_args['min_delta'], mock_args['init_delta'] = warning_min_max_delta(mock_args['min_delta'],
                                                                            mock_args['init_delta'])
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


def single_run_mock(L, dmutpb, dms, dmsp, dmsr, seed, config_number, config_run_number,
                    n_run, mock_args, data, config, validate):
    # Set the mock_args for this layer
    mock_args["L"] = L
    mock_args["delta_mutpb"] = dmutpb
    mock_args["delta_sigma"] = dms
    mock_args["delta_sigma_as_perct"] = dmsp
    mock_args["delta_inverse"] = dmsr

    # Setup some of the variables for the genotype
    MOCKGenotype.min_delta_val = mock_args['min_delta']
    MOCKGenotype.n_min_delta = MOCKGenotype.get_n_genes(mock_args['min_delta'])
    MOCKGenotype.setup_genotype_vars()
    # Setup the components class
    PartialClust.partial_clusts(
        data, mock_args["data_dict"], mock_args["argsortdists"], mock_args["L"]
    )
    # Identify the component IDs of the link origins
    MOCKGenotype.calc_reduced_clusts(mock_args["data_dict"])

    # Set the nadir point
    if validate:
        # To ensure compatible results for validation
        mock_args['hv_ref'] = [3.0, 1469.0]
    else:
        mock_args['hv_ref'] = calc_hv_ref(mock_args)

    # Strategy is not used, but kept for result consistency with adaptive
    # Same with L_comp
    for strategy, L_comp in product(
            config["strategies"], config["L_comp"]
    ):
        # Add mutation method-specific arguments
        mock_args = delta_mock.get_mutation_params(
            config["mut_method"], mock_args, L_comp
        )
        
        # Run MOCK
        start_time = time.time()
        result = delta_mock.runMOCK(seed, run_number=n_run, **mock_args)
        mp_time = time.time() - start_time
        print(f"Run {n_run} completed... It took {mp_time:.3f} secs)")

        # Extract the population
        all_pop = result[0]
        # Extract hv list
        hvs = result[1]
        # Extract final interesting links
        final_interest_inds = result[3]
        # Extract final genotype length
        final_gen_len = result[4]
        # Get the running time for each run
        time_taken = result[5]
        # Calculate the number of clusters and ARIs
        clust_aris = [evaluation.final_pop_metrics(pop, MOCKGenotype.mst_genotype, final_interest_inds, final_gen_len)
                      for pop in all_pop]
        aris = []
        num_clusts = []
        for clust_ari in clust_aris:
            num_clusts.append(clust_ari[0])
            aris.append(clust_ari[1])
        # Extract the fitness values
        var_vals = [[indiv.fitness.values[0] for indiv in pop] for pop in all_pop]
        cnn_vals = [[indiv.fitness.values[1] for indiv in pop] for pop in all_pop]
        delta_vals = [[indiv.delta for indiv in pop] for pop in all_pop]
        # Flatten list of lists
        var_vals = np.array(var_vals).flatten()
        cnn_vals = np.array(cnn_vals).flatten()
        delta_vals = np.array(delta_vals).flatten()
        aris = np.array(aris).flatten()
        num_clusts = np.array(num_clusts).flatten()

        # Generation numbers
        gen_number = []
        for i in range(1, len(all_pop)+1):
            gen_number += [i] * config['num_indivs']

        # Add strategy here for adaptive version
        n_obs = config['num_indivs'] * len(all_pop)
        results_dict = {
            "dataset": [Datapoint.data_name] * n_obs,
            'config': [config_number] * n_obs,
            'run': [config_run_number] * n_obs,
            'gen': gen_number,
            "indiv": list(range(config["num_indivs"])) * len(all_pop),
            "L": [L] * n_obs,
            "delta": delta_vals,
            "min_delta": [mock_args['min_delta']] * n_obs,
            "init_delta": [mock_args['init_delta']] * n_obs,
            "max_delta": [mock_args['max_delta']] * n_obs,
            "delta_mutpb": [dmutpb] * n_obs,
            "delta_sigma": [dms] * n_obs,
            "delta_sigma_as_perct": [dmsp] * n_obs,
            "delta_inverse": [dmsr] * n_obs,
            "VAR": var_vals,
            "CNN": cnn_vals,
            # "HV": hvs, This is evaluated per generation and not per individual
            "ARI": aris,
            "clusters": num_clusts,
            "time": [time_taken] * n_obs
        }
        # Add the new results
        results_df = pd.DataFrame(results_dict)
        hvs_df = pd.DataFrame({'N': [i for i in range(len(hvs))],
                               'config': [config_number] * len(hvs),
                               'run': [config_run_number] * len(hvs),
                               'HV': hvs})

    return results_df, hvs_df


def multi_run_mock(config, validate, name=None):
    # Load the data file paths
    if name is not None:
        print(f'Loading {name} configuration...')

    save_results = validate is False  # opposite of validate
    data_file_paths, experiment_folder = load_data(
        config["exp_name"],
        config["data_folder"],
        validate,
        config["data_subset"]
    )

    # Check the config and amend if needed
    config = utils.check_config(config)
    # Create the seed list
    seed_list, config = create_seeds(config)

    # Restrict seed_list to the actual number of runs that we need
    # Truncating like this allows us to know that the run numbers and order of seeds correspond
    seed_list = seed_list[:config["num_runs"]]
    seed_list = [(i,) for i in seed_list]  # for starmap

    # Print the number of runs to get an idea of runtime (sort of)
    print("---------------------------")
    print("Init information:")
    print(f"{config['num_runs']} run(s)")
    print(f"{len(config['strategies'])} strategy/-ies")
    print(f"{config['num_gens']} generations")
    print(f"{config['num_indivs']} individuals per generation.")
    print(f"Init delta: {config['init_delta']}.")
    print(f"Min delta: {config['min_delta']}.")
    print(f"{len(config['delta_mutation_probability'])} sigma variations")
    print(f"{len(data_file_paths)} dataset(s)")
    print("---------------------------")

    # Save the config
    utils.save_config(config, experiment_folder, validate)

    # Prepare the list of runs
    runs_list = []

    # Loop through the Ls
    n_run = 0
    config_number = 0
    for L in config["L"]:
        # Loop through delta mutation values
        for dmutpb, dms, dmsp, dmsr in zip(config["delta_mutation_probability"],
                                           config["delta_gauss_mutation_variance"],
                                           config["delta_gauss_mutation_sigma_as_perct"],
                                           config["delta_gauss_mutation_inverse"]):
            config_number += 1
            # And finally through each run
            for config_run_number, seed in enumerate(seed_list):
                n_run += 1
                runs_list.append([L, dmutpb, dms, dmsp, dmsr, seed, config_number,
                                  config_run_number + 1, n_run])

    # Loop through the data to test
    for file_path in data_file_paths:
        data_name = file_path.name

        print(f"Beginning precomputation for {data_name}...")
        # Prepare the data
        data, data_dict = prepare_data(file_path)
        # Do the precomputation for MOCK
        argsortdists, nn_rankings = setup_mock(data)
        # Wrap the arguments up for the main MOCK function
        mock_args = prepare_mock_args(data_dict, argsortdists, nn_rankings, config)

        # Decode square root deltas
        print(f'Min delta: {mock_args["min_delta"]}')
        print(f'Init delta: {mock_args["init_delta"]}')

        print("Precomputation complete!")
        print("---------------------------")

        print('Beginning {} runs.'.format(n_run))

        if n_run == 1:
            # Don't enter multiprocessing if only one run
            results = [single_run_mock(mock_args=mock_args, validate=validate,
                            data=data, config=config, *runs_list[0])]
        else:
            mock_func = partial(single_run_mock, mock_args=mock_args, validate=validate,
                            data=data, config=config)
            with multiprocessing.Pool() as pool:
                results = pool.starmap(mock_func, runs_list)
                
        results_df = pd.DataFrame()
        hvs_df = pd.DataFrame()
        for result in results:
            results_df = results_df.append(result[0], sort=False)
            hvs_df = hvs_df.append(result[1], sort=False)
        
        print(f"{file_path.name} complete!")

        # Validate the results
        if validate:
            tests.validate_results(results_df)
            print("Test passed!")
        # Save results
        if save_results:
            results_df.to_csv(f"{experiment_folder}/{data_name}_results.csv", index=False)
            hvs_df.to_csv(f"{experiment_folder}/{data_name}_hvs.csv", index=False)
            print('Results saved!')


if __name__ == '__main__':
    parser = utils.build_parser()
    cl_args = parser.parse_args()
    cl_args = vars(cl_args)
    utils.check_cl_args(cl_args)

    # Check that all config files are ok
    if cl_args['validate']:
        config_path = Path.cwd() / "configs" / "validate.json"
        configs = [utils.load_json(config_path)]
        cl_args['config'] = ['VALIDATION']
    else:
        configs = []
        for config in cl_args['config']:
            config_path = Path.cwd() / "configs" / config
            configs.append(utils.load_json(config_path))

    configs = [utils.check_config(config) for config in configs]

    # Run Algorithm
    print(f'Running {len(configs)} configuration file(s).')
    for config, name in zip(configs, cl_args['config']):
        multi_run_mock(config, cl_args['validate'], name)

    ######## TO DO ########
    # look at how results are saved and named
    # evaluation.py is a shit show
    # should clean up and define the required environment at some point
