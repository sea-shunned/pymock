import argparse
import json

def build_parser():
    parser = argparse.ArgumentParser()    

    parser.add_argument(
        '-v', '--validate',
        help='validate current setup to ensure consistent results',
        action='store_true'
    )
    parser.add_argument(
        '-e', '--exp_name',
        help='name of experiment for save location',
        type=str,
        default=""
    )
    parser.add_argument(
        '-c', '--crossover',
        help='crossover probability',
        type=float,
        default=1.0
    )
    parser.add_argument(
        '--synthdata',
        help='string to specify what subset of synthetic data to use (* selects all)',
        type=str,
        default="*"
    )
    # parser.add_argument(
    #     '--realdata',
    #     help='string to specify what subset of real data to use (* selects all)',
    #     type=str,
    #     default="*"
    # )
    parser.add_argument(
        '-m', '--mut_method',
        help='specify which mutation method to use',
        type=str,
        default="original",
        choices=["original", "centroid", "neighbour"]
    )
    parser.add_argument(
        '-runs', '--num_runs',
        help='specify the number of runs',
        type=int,
        default=30
    )
    parser.add_argument(
        '-gens', '--num_gens',
        help='specify the number of generations',
        type=int,
        default=100
    )
    parser.add_argument(
        '-indivs', '--num_indivs',
        help='specify the number of individuals in the population',
        type=int,
        default=100
    )
    parser.add_argument(
        '-L', '--L',
        help='specify the neighbourhood parameter',
        type=int,
        default=10
    )
    parser.add_argument(
        '--Lcomp',
        help='specify the component neighbourhood parameter',
        type=int,
        default=3
    )
    return parser

def check_cl_args(cl_args):
    if cl_args['validate'] and cl_args['exp_name']:
        raise ValueError("Cannot set validate and an experiment name")

    # This fails if user specifically gives default value
    # It's fine
    if cl_args['validate'] and cl_args['synthdata'] != "*":
        raise ValueError("Validate uses only 1 dataset to check consistency")

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
