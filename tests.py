import random
import os
import glob

import numpy as np

import precompute
import evaluation

def validate_results(
        results_folder, strat_name, ari_array, hv_array, fit_array, delta_triggers, num_runs):
    try:
        ari_path, fit_path, hv_path = sorted(glob.glob(results_folder+"*60*"+strat_name+"*"))
    except ValueError:
        ari_path, fit_path, hv_path, trigger_path = sorted(
            glob.glob(results_folder+"*60*"+strat_name+"*"))

    ari_orig = np.loadtxt(ari_path, delimiter=",")[:, :num_runs]
    if not np.array_equal(ari_array, ari_orig):
        print(ari_array)
        print(ari_orig)
        raise ValueError("ARI not equal")
    print("ARI correct!")
    
    fit_orig = np.loadtxt(fit_path, delimiter=",")[:num_runs*100, :]
    if not np.array_equal(fit_array, fit_orig):
        print(fit_array)
        print(fit_orig)
        raise ValueError("Fitness values not equal")
    print("Fitness correct!")

    hv_orig = np.loadtxt(hv_path, delimiter=",")[:, :num_runs]
    if not np.array_equal(hv_array, hv_orig):
        print(hv_array)
        print(hv_orig)
        raise ValueError("HV values not equal")
    print("Hypervolume correct!")

    if strat_name != "base":
        trigger_orig = [list(map(int, line.split(","))) for line in open(trigger_path)][:num_runs]
        if trigger_orig != delta_triggers:
            print(trigger_orig)
            print(delta_triggers)
            raise ValueError("Trigger generations not equal")
    print("Trigger gens correct!")
    return True
