import argparse
import json
from pathlib import Path
import warnings


def build_parser():
    parser = argparse.ArgumentParser()    
    # You need to either validate or give a config file name
    group = parser.add_mutually_exclusive_group(required=True)
    # Add the validate argument
    group.add_argument(
        '-v', '--validate',
        help='validate current setup to ensure consistent results',
        action='store_true'
    )
    # Add the config argument
    group.add_argument(
        '-c', '--config',
        help='Name of the config file (must be in configs/ subdirectory)',
        nargs='+',
        type=str
    )
    return parser


def check_cl_args(cl_args):
    # Check that the provided config exists
    if cl_args["config"] is not None:
        config_paths = [Path.cwd() / "configs" / config for config in cl_args['config']]

    # Or make sure that there is a config for validation
    elif cl_args["validate"] is not None:
        config_paths = [Path.cwd() / "configs" / "validate.json"]

    # Check that the config actually exists
    for path in config_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Config file does not exist, expected {path}")


def load_json(f_path):
    try:
        with open(f_path) as json_file:
            params = json.load(json_file)
    except json.JSONDecodeError:
        print("Unable to load config file")
        raise
    return params


def check_config(config):
    """Check that the config parameters provided are valid,
    or make them so
    """
    # Set some defaults if needed
    config = set_config_defaults(config)

    # Some of the parameters need to be lists
    list_params = ["L_comp", "delta_mutation_probability",
                   "delta_gauss_mutation_variance", "delta_gauss_mutation_sigma_as_perct",
                   "delta_gauss_mutation_inverse"]
    # If a value has been provided but isn't a list, convert it
    for key in list_params:
        if config[key] is None:
            config[key] = [None]
        elif not isinstance(config[key], list):
            config[key] = [config[key]]

    # If mutation is gauss, then sigma_perct should not be none
    if config['delta_mutation'] == 'gauss':
        if config['delta_gauss_mutation_sigma_as_perct'][0] is None or \
          config['delta_gauss_mutation_inverse'][0] is None:
            raise ValueError('"delta_gauss_mutation_sigma_as_perct" and "delta_gauss_mutation_inverse"' +
                             ' must be given for gaussian mutation')

    # Check parameter lengths are congruent
    assert len(config['delta_mutation_probability']) == len(config['delta_gauss_mutation_variance']), \
        "delta_gauss_mutation_variance and delta_mutation_probability should have the same length"
    assert len(config['delta_mutation_probability']) == len(config['delta_gauss_mutation_sigma_as_perct']), \
        "delta_gauss_mutation_sigma_as_perct and delta_mutation_probability should have the same length"
    assert len(config['delta_mutation_probability']) == len(config['delta_gauss_mutation_inverse']), \
        "delta_gauss_mutation_inverse and delta_mutation_probability should have the same length"

    return config


def set_config_defaults(config):
    # Set default values
    # Makes it easier to loop over in run_mock.run_mock()
    ## Mutation method defaults ##
    if config["mut_method"] is None:
        config["mut_method"] = "original"
        # Even if this has a value, it isn't used so override it
    if config["mut_method"] == "original":
        config["L_comp"] = [None]
    # Set a default for component-based L if not given
    elif config["mut_method"] == "centroid" or config["mut_method"] == "neighbour":
        # If we need L_comp and it isn't set, set it
        if config["L_comp"] is None: 
            # Set as a list to use product
            config["L_comp"] = [5]

    # Domain default
    if config['domain_delta'] is None:
        config['domain_delta'] = 'real'
    elif config['domain_delta'].lower() not in ['real', 'sr']:
        raise ValueError(f'{config["domain_delta"]} not implemented!')
    else:
        config['domain_delta'] = config['domain_delta'].lower()

    if config['domain_delta'] == 'sr':
        if config['delta_precision'] != 0:
            warnings.warn("Setting precision to 0...")
            config['delta_precision'] = 0

        # Standardize limits behaviour
    config['gens_step'] = 0.1
    if config['stair_limits'] is not None:
        config['flexible_limits'] = 0
        config['min_deltas'] = [100 - config['stair_limits']]
        config['max_deltas'] = [100 - 1 / 10**config['delta_precision']]
        config['gens_step'] = int(config['num_gens'] / (config['min_deltas'][0] / config['stair_limits']))
    elif config['flexible_limits'] is None or config['flexible_limits'] is False:
        config['flexible_limits'] = 0
    elif 0 < config['flexible_limits'] < 1:
        config['flexible_limits'] *= config['num_gens']

    # Check min/init delta
    if config['init_delta'] is None:
        if config['min_delta'] is None:
            warnings.warn("min delta not provided, setting to 0...")
            config['init_delta'] = 0
            config['min_delta'] = 0
        else:
            warnings.warn(f"init delta not provided, setting to {config['min_delta']}...")
            config['init_delta'] = config['min_delta']
    else:
        if config['min_delta'] is None:
            warnings.warn(f"min delta not provided, setting to {config['init_delta']}...")
            config['min_delta'] = config['init_delta']

    for var in ['min_delta', 'init_delta', 'max_delta']:
        if isinstance(config[var], str):
            if config[var][:2].lower() != 'sr':
                raise ValueError(f'{var} must be either an integer or a string starting with "sr"')

    if config['max_delta'] is None:
        config['max_delta'] = 100 - config['delta_precision']

    # Make sure crossover is well written
    if isinstance(config['crossover'], str):
        config['crossover'] = config['crossover'].lower()
        if config['crossover'] not in ['uniform']:
            raise ValueError("Crossover method '{}' not yet implemented!".format(config['crossover']))
    elif config['crossover'] is not None:
        raise ValueError("Error in crossover method '{}'".format(config['crossover']))

    # Same for delta mutation
    if isinstance(config['delta_mutation'], str):
        config['delta_mutation'] = config['delta_mutation'].lower()
        if config['delta_mutation'] not in ['gauss', 'uniform', 'random']:
            raise ValueError("Crossover method '{}' not yet implemented!".format(config['delta_mutation']))
    else:
        raise ValueError("Error in delta mutation method '{}'".format(config['delta_mutation']))

    return config


def save_config(config, experiment_folder, validate):
    """Save the final config file into the results folder for easy access
    """
    config_path = experiment_folder / f"config_{config['exp_name']}.json"

    if validate is None:
        with open(config_path, "w") as out_file:
            json.dump(config, out_file, indent=4)
