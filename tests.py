from pathlib import Path

import numpy as np

def validate_results(strategy, ari_array, hv_array, fit_array, delta_triggers, num_runs):
    # Set the validation results folder path
    results_folder = Path.cwd() / "validation" / "results"
    # Extract the paths for the relevant measures
    ari_path, fit_path, hv_path = sorted(list(results_folder.glob("*60*"+strategy+"*")))
    # Evaluate the ARI
    ari_orig = np.loadtxt(ari_path, delimiter=",")[:, :num_runs]
    if not np.array_equal(ari_array, ari_orig):
        print("Original, then current")
        print(np.hstack((ari_orig, ari_array)))
        raise ValueError("ARI not equal")
    print("ARI correct!")
    # Evaluate the fitness
    fit_orig = np.loadtxt(fit_path, delimiter=",")[:num_runs*100, :]
    if not np.array_equal(fit_array, fit_orig):
        print("Original, then current")
        print(np.hstack((fit_orig, fit_array)))
        raise ValueError("Fitness values not equal")
    print("Fitness correct!")
    # Evaluate the HV
    hv_orig = np.loadtxt(hv_path, delimiter=",")[:, :num_runs]
    if not np.array_equal(hv_array, hv_orig):
        print("Original, then current")
        print(np.hstack((hv_orig, hv_array)))
        raise ValueError("HV values not equal")
    print("Hypervolume correct!")
    # Evaluate the triggers
    if strategy != "base":
        trigger_orig = [list(map(int, line.split(","))) for line in open(trigger_path)][:num_runs]
        if trigger_orig != delta_triggers:
            print(trigger_orig)
            print(delta_triggers)
            raise ValueError("Trigger generations not equal")
    print("Trigger gens correct!")
    return True

def validate_mock(results_df, delta_triggers, strategy, num_runs):
    # Get the expected arrays from the dataframe
    # This is for historical reasons
    ari_array, hv_array, fit_array = convert_df(results_df)

    valid = validate_results(
        strategy, ari_array, hv_array, fit_array,
        delta_triggers, num_runs
    )
    return valid

def convert_df(results_df):
    ari_df = results_df.pivot(index="indiv", columns="run", values="ARI")
    ari_array = ari_df.values
    
    hv_df = results_df.pivot(index="indiv", columns="run", values="HV")
    hv_array = hv_df.values

    fit_df = results_df[["VAR", "CNN", "run"]]
    fit_array = fit_df.values

    return ari_array, hv_array, fit_array