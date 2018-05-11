# Have a single script that I can use to run MOCK a couple of times with different configs and compare results to ensure that the results I get are the same

import random
import os
import glob
import numpy as np
import precompute
import main_base

def loadData():
    basepath = os.getcwd()

    data_folder = os.path.join(basepath, "data")+os.sep
    synth_data_folder = os.path.join(data_folder, "synthetic_datasets")+os.sep
    synth_data_files = glob.glob(synth_data_folder+'tevc_20_60_9*.data')
    results_folder = basepath+"/results/"
    data_files = synth_data_files

    # add assert here for data_name? Need to make sure we're using a particular dataset

    return data_files

def loadArtifs():
    # import main_base # for DEAP indiv declaration
    import artif_carryon
    import artif_hypermutspec
    import artif_hypermutall
    import artif_reinit
    import artif_fairmut

    funcs = [artif_carryon.main, artif_hypermutspec.main, artif_hypermutall.main, artif_reinit.main, artif_fairmut.main]
    return funcs

def loadMains():
    pass
    # import main_base # for DEAP indiv declaration
    import main_carryon
    import main_hypermutspec
    import main_hypermutall
    import main_reinit
    import main_fairmut

    funcs = [main_carryon.main, main_hypermutspec.main, main_hypermutall.main, main_reinit.main, main_fairmut.main]
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

    results_folder_data = results_folder+classes.Dataset.data_name+os.sep

    # Add square root delta values
    delta_val = 100-((100*sr_val*np.sqrt(classes.Dataset.num_examples))/classes.Dataset.num_examples)

    distarray = precompute.compDists(data, data)
    distarray = precompute.normaliseDistArray(distarray)

    argsortdists = np.argsort(distarray, kind='mergesort')
    nn_rankings = precompute.nnRankings(distarray, classes.Dataset.num_examples)
    mst_genotype = precompute.createMST(distarray)
    degree_int = precompute.degreeInterest(mst_genotype, L, nn_rankings, distarray)
    int_links_indices = precompute.interestLinksIndices(degree_int)
    print("Precomputation done!\n")

    HV_ref = None    

    args = data, data_dict, delta, HV_ref, argsortdists, nn_rankings, mst_genotype, int_links_indices, L, num_indivs, num_gens, delta_reduce
    return args



def runMOCK(file_path, funcs):
    fitness_cols = ["VAR", "CNN", "Run"]
    args = prepareArgs(file_path)

    for func in funcs:
        strat_name = func.__globals__["__file__"].split("/")[-1].split(".")[0].split("_")[-1]

        # Create arrays to save results for the given function
        fitness_array = np.empty((num_indivs, len(fitness_cols)))
        hv_array = np.empty((num_gens, 1))
        ari_array = np.empty((num_indivs, 1))
        delta_triggers = []

        random.seed(11)

        pop, HV, HV_ref_temp, int_links_indices_spec, relev_links_len, adapt_gens = func(*args)

        fitness_array[ind:ind+num_indivs,0:3] = [indiv.fitness.values+(run+1,) for indiv in pop]

        _, aris = evaluation.finalPopMetrics(pop, mst_genotype, int_links_indices_spec, relev_links_len)

        ari_array[:, 1] = aris
        hv_array[:, run] = HV
        delta_triggers.append(adapt_gens)

        valid = validateResults(strat_name, ari_array, hv_array, delta_triggers)

        if not valid:
            raise ValueError(f"Results incorrect for {strat_name}")

def validateResults(strat_name, ari_array, hv_array, delta_triggers):
    # Take the hypervolume and/or ARI results generated and compare them to a saved version of the results for each strategy to ensure it's the same
    pass

def main():
    data_files = loadData()

    return data_files

    funcs = [main_base.main]
    funcs.extend(loadMains())

    for file_path in data_files:
        runMOCK(file_path, funcs)

    ## Artif scripts are for the random or interval triggers, which we're unlikely to use in the future
    # funcs = [main_base.main]
    # funcs.extend(loadArtifs())

    # for file_path in data_files:
    #     runMOCK(file_path, funcs)



############# To Do #############
# Possible to delete MST/graph stuff once we get the MST genotype? Free some memory?

if __name__ == '__main__':
    print(main())