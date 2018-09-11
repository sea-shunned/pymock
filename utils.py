import argparse

def build_parser():
    parser = argparse.ArgumentParser()

    ####### TO DO
    # consider adding the following arguments:
        # L
        # num_gens
        # num_indivs
    # basically what from config should actually be here, and what should stay in config
    # probably single numbers and easy stuff goes here
    

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
        default="original"
    )
    return parser

def check_cl_args(cl_args):
    if cl_args['validate'] and cl_args['exp_name']:
        raise ValueError("Cannot set validate and an experiment name")

    if cl_args['validate'] and cl_args['synthdata'] != "*":
        raise ValueError("Validate uses only 1 dataset to check consistency")