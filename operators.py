import random
from numpy.random import randint
from classes import MOCKGenotype


def uniform_xover(parent1, parent2, cxpb):
    """Uniform crossover
    
    Arguments:
        parent1 {DEAP individual} -- First parent
        parent2 {DEAP individual} -- Second parent
        cxpb {float} -- Probability of undergoing crossover
    
    Returns:
        [type] -- [description]
    """
    # xover only the minimum length
    target_len = min(len(parent1), len(parent2))

    # Make copies of the parents
    child1 = parent1[:]
    child2 = parent2[:]

    # Test if we undergo crossover
    if cxpb == 1:
        for i in range(target_len):
            if random.random() < 0.5:
                parent1[i] = child1[i]
                parent2[i] = child2[i]
            else:
                parent1[i] = child2[i]
                parent2[i] = child1[i]
        # Caveat: there is a 0.5**len(parent1) chance the parents will be equal to children
        del parent1.fitness.values, parent2.fitness.values

    # In case another probability is used; we avoid a random.random() call in normal case
    elif random.random() <= cxpb:
        for i in range(target_len):
            if random.random() < 0.5:
                parent1[i] = child1[i]
                parent2[i] = child2[i]
            else:
                parent1[i] = child2[i]
                parent2[i] = child1[i]
        del parent1.fitness.values, parent2.fitness.values
    # If we change cxpb to be <1 then we may not enter loop, so return unchanged
    # We'll keep their fitnesses so we don't need to re-evaluate (unless mutation changes)
    return parent1, parent2


def no_xover(parent1, parent2):
    return parent1, parent2


def neighbour_mut(parent, MUTPB, argsortdists, L, interest_indices, nn_rankings):
    """Neighbourhood-biased mutation operator
    
    Arguments:
        parent {DEAP individual} -- The genotype
        MUTPB {float} -- Mutation probability threshold
        gen_length {int} -- Equal to reduced_length when delta>0, or N otherwise
        argsortdists {np.array} -- Argsort of the distance array
        L {int} -- Neighbourhood hyperparameter
        interest_indices {list} -- Indices of the most interesting links
        nn_rankings {np.array} -- Nearest neighbour rankings for every datapoint
    
    Returns:
        parent {DEAP individual} -- The mutated genotype
    """
    if len(parent) > 0:
        # Calculate the first term of the mutation equation
        first_term = (MUTPB / len(parent))
        # Using a comprehension for this bit is faster
        mutprobs = [
            first_term +
            ((nn_rankings[interest_indices[index]][value] / len(parent)) ** 2)
            for index, value in enumerate(parent)
        ]
        # Now just loop over the probabilities
        for index, mutprob in enumerate(mutprobs):
            if random.random() < mutprob:
                parent[index] = MOCKGenotype.replace_link(
                    argsortdists, interest_indices[index], parent[index], L
                )

    return parent


def comp_centroid_mut(parent, MUTPB, argsortdists_cen, L_comp, interest_indices, nn_rankings_cen, data_dict):
    if len(parent) > 0:
        first_term = (MUTPB / len(parent))
        # Using a comprehension for this bit is faster
        mutprobs = [
            first_term +
            ((nn_rankings_cen[data_dict[index].base_cluster_num][data_dict[value].base_cluster_num] / len(parent)) ** 2)
            for index,value in enumerate(parent)
        ]
        # Now just loop over the probabilities
        for index, mutprob in enumerate(mutprobs):
            if random.random() < mutprob:
                parent[index] = MOCKGenotype.centroid_replace_link(
                    argsortdists_cen, interest_indices[index], parent[index], L_comp, data_dict
                )

    return parent


def neighbour_comp_mut(parent, MUTPB, interest_indices, nn_rankings, component_nns, data_dict):
    if len(parent) > 0:
        first_term = (MUTPB / len(parent))
        mutprobs = [
            first_term +
            ((nn_rankings[interest_indices[index]][value] / len(parent)) ** 2)
            for index,value in enumerate(parent)
        ]
        for index, mutprob in enumerate(mutprobs):
            if random.random() < mutprob:
                parent[index] = MOCKGenotype.neighbour_replace_link(
                    component_nns, interest_indices[index], parent[index], data_dict
                )

    return parent


def gaussian_mutation_delta(parent, sigma, MUTPB, min_delta, max_delta, gen_n, precision=3,
                            sigma_perct=False, inverse=False, flexible_limits=0, bias=0):
    # Set parameters
    mu = parent.delta + bias
    if sigma_perct:
        if inverse:
            sigma = (100-mu)*sigma + 1/(101-mu)
        else:
            sigma = mu * sigma + (100-mu)/100

    # Test if mutation is happening
    if random.random() < MUTPB:
        old_delta = parent.delta

        # Mutate
        new_delta = random.gauss(mu, sigma)
        if gen_n >= flexible_limits:
            # Flexible limits means that no hard min/max limits are imposed on delta
            parent.delta = round(max(min(new_delta, 100), 0), precision)
        else:
            parent.delta = round(max(min(new_delta, max_delta), min_delta), precision)

        # Update its genotype
        parent = MOCKGenotype.expand_reduce_genotype(parent, old_delta)

    return parent


def random_delta(parent, min_delta, max_delta, precision):
    old_delta = parent.delta

    # Change the delta of the parent to a new random number
    new_delta = randint(min_delta*10**precision, max_delta*10**precision+precision)
    parent.delta = new_delta*10**precision

    # Update its genotype
    parent = MOCKGenotype.expand_reduce_genotype(parent, old_delta)

    return parent
