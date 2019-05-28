import argparse
import json
from pathlib import Path

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
        type=str
    )
    return parser

def check_cl_args(cl_args):
    # Check that the provided config exists
    if cl_args["config"] is not None:
        config_path = Path.cwd() / "configs" / cl_args["config"]
    # Or make sure that there is a config for validation
    elif cl_args["validate"] is not None:
        config_path = Path.cwd() / "configs" / "validate.json"
    # Check that the config actually exists
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file does not exist, expected {config_path}")

def load_json(f_path):
    try:
        with open(f_path) as json_file:
            params = json.load(json_file)
    except JSONDecodeError:
        print("Unable to load config file")
        raise
    return params

def check_config(config):
    """Check that the config parameters provided are valid,
    or make them so
    """
    if config["delta_sr_vals"] is None and config["delta_raw_vals"] is None:
        raise ValueError(f"A delta value must be provided")
    # Set some defaults if needed
    config = set_config_defaults(config)
    # Some of the parameters need to be lists
    list_params = ["delta_sr_vals", "delta_raw_vals", "L_comp"]
    # If a value has been provided but isn't a list, convert it
    for key in list_params:
        if config[key] is not None and not isinstance(config[key], list):
            config[key] = list(config[key])
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
    return config

def save_config(config, experiment_folder, validate):
    """Save the final config file into the results folder for easy access
    """
    config_path = experiment_folder / f"config_{config['exp_name']}.json"

    if validate is None:
        with open(config_path, "w") as out_file:
            json.dump(config, out_file, indent=4)
