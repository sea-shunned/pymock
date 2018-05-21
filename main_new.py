import precompute
import initialisation
import objectives
import operators
import classes

# For multiprocessing
from os import cpu_count
import multiprocessing

from deap import base, creator, tools
from deap.benchmarks.tools import hypervolume

import random


def delta_precomp(delta_val, mst_genotype, int_links_indices, ):
    relev_links_len = initialisation.relevantLinks(
        delta_val, classes.Dataset.num_examples)
    print("Genotype length:", relev_links_len)

    base_genotype, base_clusters = initialisation.baseGenotype(
        mst_genotype, int_links_indices, relev_links_len)

    classes.partialClustering(
        base_clusters, data, data_dict, argsortdists, L)

    # Maybe also put this as a class attribute for PartialClust?
    reduced_clust_nums = [data_dict[i].base_cluster_num for i in int_links_indices[:relev_links_len]]

# Consider trying to integrate the use of **kwargs here
def create_base_toolbox(num_indivs, mst_genotype, int_links_indices, relev_links_len, argsortdists, L, data_dict, nn_rankings):
    toolbox = base.Toolbox()

    # Register the initialisation function
    toolbox.register(
        "initDelta", 
        initialisation.initDeltaMOCK, 
        k_user = classes.Dataset.k_user, 
        num_indivs = num_indivs, 
        mst_genotype = mst_genotype, 
        int_links_indices = int_links_indices, 
        relev_links_len = relev_links_len, 
        argsortdists = argsortdists, 
        L = L
        )

    # Register the population function that uses the custom initialisation
    toolbox.register(
        "population", 
        tools.initIterate, 
        list, 
        toolbox.initDelta
        )

    # Register the evaluation function
    toolbox.register(
        "evaluate", 
        objectives.evalMOCK, 
        part_clust = classes.PartialClust.part_clust, 
        reduced_clust_nums = reduced_clust_nums, 
        conn_array = classes.PartialClust.conn_array, 
        max_conn = classes.PartialClust.max_conn, 
        num_examples = classes.Dataset.num_examples, 
        data_dict = data_dict, 
        cnn_pairs = classes.PartialClust.cnn_pairs, 
        base_members = classes.PartialClust.base_members, 
        base_centres = classes.PartialClust.base_centres
        )

    # Register the crossover function
    toolbox.register(
        "mate", 
        operators.uniformCrossover, 
        cxpb = 1.0
        )

    # Register the mutation function
    toolbox.register(
        "mutate", operators.neighbourMutation, 
        MUTPB = 1.0, 
        gen_length = relev_links_len, 
        argsortdists = argsortdists, 
        L = L, 
        int_links_indices = int_links_indices, 
        nn_rankings = nn_rankings
        )
    
    # Register the selection function (built-in with DEAP for NSGA2)
    toolbox.register(
        "select", 
        tools.selNSGA2
        )
    
    return toolbox

def generation(pop, toolbox, HV, HV_ref):
    # Shuffle the population
    random.shuffle(pop)

    # Select and clone the offspring (to allow modification)
    offspring = tools.selTournamentDCD(pop, len(pop))
    offspring = toolbox.map(toolbox.clone, offspring)

    # Tournament
    for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
        # Crossover
        toolbox.mate(ind1, ind2)

        # Mutation
        toolbox.mutate(ind1)
        toolbox.mutate(ind2)

    # Evaluate the offspring
    fitnesses = toolbox.map(toolbox.evaluate, offspring)
    for ind, fit in zip(offspring, fitnesses):
        ind.fitness.values = fit

    # Select from the current population and new offspring
    pop = toolbox.select(pop + offspring, num_indivs)

    HV.append(hypervolume(pop, HV_ref))

def initial_setup(toolbox):
    pop = toolbox.population()

	# Convert each individual of class list to class deap.creator.Individual
	# Easier than modifying population function
    for index, indiv in enumerate(pop):
        indiv = creator.Individual(indiv)
        pop[index] = indiv

    # Lists to capture the initial population fitness (if desired)
    VAR_init = []
    CNN_init = []

    # Evaluate the initial pop
    fitnesses = toolbox.map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit	
        VAR_init.append(fit[0])
        CNN_init.append(fit[1])
    
    # This is just to assign the crowding distance to the individuals, no actual selection is done
    pop = toolbox.select(pop, len(pop))

    return pop, VAR_init, CNN_init

def calc_HV_ref(VAR_init):
    return [np.ceil(np.max(VAR_init)*1.5), np.ceil(classes.PartialClust.max_conn+1)]

def main(**kwargs):
    toolbox = create_base_toolbox()

    pop, VAR_init, CNN_init = initial_setup(toolbox)

    if HV_ref == None:
        HV_ref = calc_HV_ref(pop)

    # Check that the HV_ref reference point is valid
    if classes.PartialClust.max_conn >= HV_ref[1]:
        raise ValueError(f"Max CNN value ({classes.PartialClust.max_conn}) has exceeded that set for HV reference point ({HV_ref[1]}); HV values may be unreliable")

    for gen in range(1, num_gens):
        pop = generation(pop, toolbox)