from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs
import numpy as np
import igraph
import classes
import precompute
import evaluation
import initialisation
import objectives
import delta_mock_mp
import run_mock_mp

from itertools import count


np.random.seed(42)

params = run_mock_mp.load_config(config_path="mock_config.json")

classes.Dataset.num_examples = 16
classes.Dataset.num_features = 10
classes.Dataset.k_user = 3
L = 10
delta_reduce = 1
num_indivs = 100
num_gens = 100

data, labels = make_blobs(n_samples=classes.Dataset.num_examples, centers=3, n_features=10)
classes.Dataset.labels = False
classes.Dataset.label_vals = labels

_, data_dict = classes.Dataset.createDatasetGarza(data)

# Go through the precomputation specific to the dataset
distarray = precompute.compDists(data, data)
distarray = precompute.normaliseDistArray(distarray)
argsortdists = np.argsort(distarray, kind='mergesort')
nn_rankings = precompute.nnRankings(distarray, classes.Dataset.num_examples)
classes.MOCKGenotype.mst_genotype = precompute.createMST(distarray)
classes.MOCKGenotype.degree_int = precompute.degreeInterest(
    classes.MOCKGenotype.mst_genotype, L, nn_rankings, distarray)

# Finds the indices of the most to least interesting links
classes.MOCKGenotype.interest_links_indices()

kwargs = {
    "data": data,
    "data_dict": data_dict,
    "delta_val": None,
    "hv_ref": None,
    "argsortdists": argsortdists,
    "nn_rankings": nn_rankings,
    # "mst_genotype": mst_genotype,
    # "int_links_indices": int_links_indices,
    "L": L,
    "num_indivs": num_indivs,
    "num_gens": num_gens,
    "delta_reduce": delta_reduce,
    "strat_name": None,
    "adapt_delta": None,
    "relev_links_len": None,
    "reduced_clust_nums": None
    # "seed_num": None
}

seed_list = [(1,)]
sr_vals = [1]

print("Scikit-learn 1 cluster inertia:")
print(KMeans(n_clusters=1).fit(data).inertia_, "\n")

for sr_val in sr_vals:
    # kwargs['delta_val'] = 100-(
    #     (100*sr_val*np.sqrt(classes.Dataset.num_examples))/classes.Dataset.num_examples
    #     )
    
    classes.MOCKGenotype.calc_delta(sr_val)

    classes.MOCKGenotype.setup_genotype_vars()

    classes.PartialClust.id_value = count()
    # relev_links_len, reduced_clust_nums = run_mock_mp.delta_precomp(
    #     kwargs['data'], kwargs["data_dict"], kwargs["argsortdists"], kwargs["L"], kwargs['delta_val'], 
    #     kwargs["mst_genotype"], kwargs["int_links_indices"]
    #     )
    # kwargs["relev_links_len"] = relev_links_len
    # kwargs["reduced_clust_nums"] = reduced_clust_nums
    # print(reduced_clust_nums, "red_clust_nums")
    # print(kwargs["int_links_indices"], "int_links_indices")

    print(classes.MOCKGenotype.degree_int)
    print(classes.MOCKGenotype.base_genotype)
    print(classes.MOCKGenotype.base_clusters)
    print(classes.MOCKGenotype.reduced_genotype_indices)
    print(classes.MOCKGenotype.mst_genotype)

    classes.PartialClust.partial_clusts(data, data_dict, argsortdists, L)
    
    classes.MOCKGenotype.calc_reduced_clusts(data_dict)
    print(classes.MOCKGenotype.reduced_cluster_nums)

    raise

    mst_reduced_genotype = [kwargs['mst_genotype'][i] for i in kwargs['int_links_indices'][:relev_links_len]]
    print(kwargs['int_links_indices'][:relev_links_len], "relevant link indices")
    print(mst_reduced_genotype, "reduced mst genotype")
    print(kwargs['mst_genotype'], "original mst genotype")
    chains, superclusts = objectives.clusterChains(
        mst_reduced_genotype, kwargs['data_dict'], classes.PartialClust.part_clust, reduced_clust_nums)
    print(superclusts, "superclusts")
    classes.PartialClust.max_var = objectives.objVAR(
        chains, classes.PartialClust.part_clust, classes.PartialClust.base_members,
        classes.PartialClust.base_centres, superclusts
    )
    print("Max VAR:", classes.PartialClust.max_var)

    # raise

    pop, hv, hv_ref, int_links_indices, relev_links_len, adapt_gens= delta_mock_mp.runMOCK(*list(kwargs.values()), 1)

    temp_res = sorted([ind.fitness.values for ind in pop], key=lambda var:var[1])
    # print(temp_res[:2])
    # print(temp_res[-2:])

    # print([ind.fitness.values for ind in pop])
    print(kwargs['mst_genotype'])
    pop = sorted(pop, key=lambda var:var.fitness.values[1])
    print(pop[0], pop[0].fitness.values)

    # with multiprocessing.Pool() as pool:
    #     results = pool.starmap(mock_func, seed_list)
    

    g = igraph.Graph()
    g.add_vertices(len(kwargs['mst_genotype']))
    g.add_edges(zip(range(len(kwargs['mst_genotype'])),kwargs['mst_genotype']))
    print(len(list(g.components(mode='WEAK'))))

    if len(pop[0]) == len(kwargs["mst_genotype"]):
        h = igraph.Graph()
        h.add_vertices(len(kwargs['mst_genotype']))
        h.add_edges(zip(range(len(kwargs['mst_genotype'])),pop[0]))
        print(len(list(h.components(mode='WEAK'))))