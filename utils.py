import warnings


def check_config(c):
    """Check that the config parameters provided are valid, or
    make them so. Parameter c is an instance of class DMOCK."""

    # Set some defaults if needed
    set_config_defaults(c)

    # If mutation is gauss, then sigma_perct should not be none
    if c.delta_mutation == 'gauss' and (c.delta_as_perct is None or c.delta_inverse is None):
        raise ValueError('"delta_as_perct" and "delta_inverse" must be given for gaussian mutation')

    # Check legal values were set
    assert c.domain_delta in ['real', 'sr'], f'{c.domain_delta} not implemented!'
    assert c.crossover in ['uniform'], f'Crossover method "{c.crossover}" not implemented!'

    if isinstance(c.delta_mutation, str):
        c.delta_mutation = c.delta_mutation.lower()
        if c.delta_mutation not in ['gauss', 'uniform', 'random']:
            raise ValueError(f"Crossover method '{c.delta_mutation}' not yet implemented!")
    else:
        raise ValueError(f"Error in delta mutation method '{c.delta_mutation}'")


def set_config_defaults(c):
    """Set default values. Parameter c is an instance of class DMOCK."""

    # L_comp is not used in original, so override it
    if c.mut_method == "original":
        c.L_comp = None
    # Set a default for component-based L if not given
    elif c.mut_method in ["centroid", "neighbour"]:
        # If we need L_comp and it isn't set, set it
        if c["L_comp"] is None:
            # Set as a list to use product
            c.L_comp = [5]

    # Domain default
    if c.domain_delta == 'sr':
        if c.delta_precision != 0:
            warnings.warn("Setting precision to 0...")
            c.delta_precision = 0

    # Standardize limits behaviour
    if c.stair_limits is not None:
        # Overide flexible_limits behaviour
        c.flexible_limits = 0
        c.min_deltas = [100 - c.stair_limits]
        c.max_deltas = [100 - (1 / (10 ** c.delta_precision))]
        c.gens_step = int(c.num_gens / (c.min_deltas[0] / c.stair_limits))

    elif c.flexible_limits is None or c.flexible_limits is False:
        c.flexible_limits = 0

    elif 0 < c.flexible_limits < 1:
        # If percentual, calculate the number of generations it represents
        c.flexible_limits *= c.num_gens

    # Check min/init delta
    if c.init_delta is None:
        if c.min_delta is None:
            warnings.warn("min delta not provided, setting to 0...")
            c.init_delta = 0
            c.min_delta = 0
        else:
            warnings.warn(f"init delta not provided, setting to {c.min_delta}...")
            c.init_delta = c.min_delta
    else:
        if c.min_delta is None:
            warnings.warn(f"min delta not provided, setting to {c.init_delta}...")
            c.min_delta = c.init_delta

    if c.max_delta is None:
        c.max_delta = 100 - c.delta_precision

    for var in [c.min_delta, c.init_delta, c.max_delta]:
        if isinstance(var, str):
            if var[:2].lower() != 'sr':
                raise ValueError(f'All delta limits must be either an integer or a string starting with "sr"')
