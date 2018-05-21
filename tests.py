import random
import os
import glob
import numpy as np
import precompute, evaluation
import main_base

def loadData():
    basepath = os.getcwd()

    data_folder = os.path.join(basepath, "data")+os.sep
    synth_data_folder = os.path.join(data_folder, "synthetic_datasets")+os.sep
    synth_data_files = glob.glob(synth_data_folder+'tevc_20_60_9*.data')
    results_folder = basepath+"/test_data/"
    data_files = synth_data_files

    # add assert here for data_name? Need to make sure we're using a particular dataset

    return data_files, results_folder

def loadArtifs():
    import artif_carryon
    import artif_hypermutspec
    import artif_hypermutall
    import artif_reinit
    # import artif_fairmut

    funcs = [
        artif_carryon.main, artif_hypermutspec.main, 
        artif_hypermutall.main, artif_reinit.main]
    return funcs

def loadMains():
    import main_carryon
    import main_hypermutspec
    import main_hypermutall
    import main_reinit
    # import main_fairmut # inconsistent between runs with same seed

    funcs = [
        main_carryon.main, main_hypermutspec.main, 
        main_hypermutall.main, main_reinit.main]
    return funcs

def prepareArgs(file_path, L=10, num_indivs=100, num_gens=100, sr_val=1, delta_reduce=1):
    import classes

    classes.Dataset.data_name = file_path.split("/")[-1].split(".")[0][:-15]

    with open(file_path) as file:
        head = [int(next(file)[:-1]) for _ in range(4)]

    # Read the data into an array
    data = np.genfromtxt(file_path, delimiter="\t", skip_header=4)

    # Set the values for the data
    classes.Dataset.num_examples = head[0] # Num examples
    classes.Dataset.num_features = head[1] # Num features/dimensions
    classes.Dataset.k_user = head[3] # Num real clusters

    # Do we have labels?
    if head[2] == 1:
        classes.Dataset.labels = True
    else:
        classes.Dataset.labels = False

    # Remove labels if present and create data_dict
    data, data_dict = classes.createDatasetGarza(data)

    # Add square root delta values
    delta = 100-((100*sr_val*np.sqrt(classes.Dataset.num_examples))/classes.Dataset.num_examples)

    print(f"Testing delta = {delta}")

    distarray = precompute.compDists(data, data)
    distarray = precompute.normaliseDistArray(distarray)

    argsortdists = np.argsort(distarray, kind='mergesort')
    nn_rankings = precompute.nnRankings(distarray, classes.Dataset.num_examples)
    mst_genotype = precompute.createMST(distarray)
    degree_int = precompute.degreeInterest(mst_genotype, L, nn_rankings, distarray)
    int_links_indices = precompute.interestLinksIndices(degree_int)
    print("Precomputation done!\n")

    # Hard-coded HV_ref to this dataset so that we can compare and make sure the HV is right
    HV_ref = [3.0, 1469.0]    

    args = data, data_dict, delta, HV_ref, argsortdists, nn_rankings,mst_genotype, int_links_indices, L, num_indivs, num_gens, delta_reduce
    return args, mst_genotype

def runMOCK(file_path, funcs, results_folder):
    fitness_cols = ["VAR", "CNN", "Run"]
    args, mst_genotype = prepareArgs(file_path)

    num_indivs = 100
    num_gens = 100

    for func in funcs:
        strat_name = func.__globals__["__file__"].split("/")[-1].split(".")[0].split("_")[-1]
        print(f"Testing {strat_name}")

        # Create arrays to save results for the given function
        fit_array = np.empty((num_indivs, len(fitness_cols)))
        # hv_array = np.empty((num_gens, 1))
        ari_array = np.empty((num_indivs, 1))
        delta_triggers = []

        random.seed(11)

        pop, HV, _, int_links_indices_spec, relev_links_len, adapt_gens = func(*args)

        fit_array[:num_indivs,0:3] = [indiv.fitness.values+(1,) for indiv in pop]

        _, aris = evaluation.finalPopMetrics(
            pop, mst_genotype, int_links_indices_spec, relev_links_len)

        # ari_array[:, 0] = aris
        # hv_array[:, 0] = HV
        delta_triggers.append(adapt_gens)

        valid = validateResults(results_folder, strat_name, np.asarray(aris), np.asarray(HV), fit_array, delta_triggers)

        if not valid:
            raise ValueError(f"Results incorrect for {strat_name}")

        else:
            print(f"{strat_name} validated!\n")

def validateResults(
    results_folder, strat_name, ari_array, hv_array, fit_array, delta_triggers):
    # Take the hypervolume and/or ARI results generated and compare them to a saved version of the results for each strategy to ensure it's the same
    if strat_name != "base":
        ari_path, fit_path, hv_path, trigger_path = sorted(glob.glob(results_folder+"*"+strat_name+"*"))
    else:
        ari_path, fit_path, hv_path = sorted(glob.glob(results_folder+"*"+strat_name+"*"))

    ari_orig = np.loadtxt(ari_path, delimiter=",")[:, 0]
    if not np.array_equal(ari_array, ari_orig):
        print(ari_array)
        print(ari_orig)
        raise ValueError("ARI not equal")

    fit_orig = np.loadtxt(fit_path, delimiter=",")[:100, :]
    if not np.array_equal(fit_array, fit_orig):
        raise ValueError("Fitness values not equal")

    # Only useful if we have fixed the same HV ref point as the orig experiments (which we have)
    hv_orig = np.loadtxt(hv_path, delimiter=",")[:, 0]
    if not np.array_equal(hv_array, hv_orig):
        raise ValueError("HV values not equal")

    if strat_name != "base":
        trigger_orig = [list(map(int, line.split(","))) for line in open(trigger_path)][0]
        if trigger_orig != delta_triggers[0]:
            raise ValueError("Trigger generations not equal")

    return True

def main():
    data_files, results_folder = loadData()

    # validateResults(results_folder, 'carryon', None, None, None)

    funcs = [main_base.main]
    funcs.extend(loadMains())

    for file_path in data_files:
        runMOCK(file_path, funcs, results_folder)

    ## Artif scripts are for the random or interval triggers, which we're unlikely to use in the future
    # funcs = [main_base.main]
    # funcs.extend(loadArtifs())

    # for file_path in data_files:
    #     runMOCK(file_path, funcs)



############# To Do #############
# Possible to delete MST/graph stuff once we get the MST genotype? Free some memory?
# Have a single script that I can use to run MOCK a couple of times with different configs and compare results to ensure that the results I get are the same

if __name__ == '__main__':
    main()