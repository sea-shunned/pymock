import warnings


def warning_min_max_delta(min_delta, max_delta):
    if min_delta > max_delta:
        warnings.warn('Swapping min and max delta...', Warning)
        min_delta, max_delta = max_delta, min_delta
    if max_delta > 100:
        warnings.warn('Setting max_delta to 100...', Warning)
        max_delta = 100
    if min_delta < 0:
        warnings.warn('Setting min_delta to 0...', Warning)
        max_delta = 0
    return min_delta, max_delta