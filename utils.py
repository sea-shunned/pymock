import argparse
import json
from pathlib import Path

def build_parser():
    parser = argparse.ArgumentParser()    
    # You need to either validate or give a config file name
    group = parser.add_mutually_exclusive_group(require=True)
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
    # Create the path to the config file to check it
    config_path = Path.cwd() / "configs" / cl_args["config"]
    # Check that the config actually exists
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file does not exist, expected {config_path}")

def load_json(f_path="mock_config.json"):
    """Load config file for MOCK
    
    Keyword Arguments:
        f_path {str} -- [description] (default: {"mock_config.json"})
    
    Returns:
        [type] -- [description]
    """
    try:
        with open(f_path) as json_file:
            params = json.load(json_file)
    except JSONDecodeError:
        print("Unable to load config file")
        raise
    return params
