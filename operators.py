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


def gaussian_mutation_delta(parent, sigma, mutpb, min_sr, init_sr, min_delta,  init_delta, max_delta, gen_n,
                            precision=3, sigma_perct=False, inverse=False, flexible_limits=0, squash=False):
    # Test if mutation is happening
    sr = min_sr is not None
    if random.random() < mutpb:
        # Set parameters
        if sr:
            mu = parent.sr
            ub = 2 * mu
            min_sigma = 2/3
        else:
            mu = parent.delta
            ub = 100
            min_sigma = 0.1

        if sigma_perct:
            if inverse:
                sigma = (ub - mu) * sigma + 1 / (ub - mu + 1)
            else:
                sigma = mu * sigma + (ub - mu) / ub

        if squash:
            sigma = min(min(sigma, (ub-mu)/3), mu/3)
            if sigma <= min_sigma:
                sigma = min_sigma

        old_delta = parent.delta

        # Mutate
        if sr:
            new_sr = random.gauss(mu, sigma)

            # Flexible limits means that no hard min/max limits are imposed on delta
            if gen_n >= flexible_limits:
                new_sr = min_max(new_sr, 1, min_sr, precision)
            else:
                new_sr = min_max(new_sr, 1, init_sr, precision)

            new_delta = MOCKGenotype.sr_vals[new_sr]
        else:
            new_sr = None
            new_delta = random.gauss(mu, sigma)

            if gen_n >= flexible_limits:
                new_delta = min_max(new_delta, min_delta, max_delta, precision)
            else:
                new_delta = min_max(new_delta, init_delta, max_delta, precision)

        parent.sr = int(new_sr)
        parent.delta = new_delta

        # Update its genotype
        parent = MOCKGenotype.expand_reduce_genotype(parent, old_delta)

    return parent


def uniform_mutation_delta(parent, spread, mutpb, min_sr, init_sr, min_delta, init_delta, max_delta, gen_n, precision=3,
                           flexible_limits=0, squash=False):
    sr = min_sr is not None
    if random.random() < mutpb:
        # Set parameters
        if sr:
            mu = parent.sr
            ub = 2 * mu
            min_diff = 1
        else:
            mu = parent.delta
            ub = 100
            min_diff = 0.1

        if gen_n >= flexible_limits:
            init_delta = min_delta
            max_delta = 100
            init_sr = min_sr

        if squash:
            diff = min(min(spread, ub-mu), mu)
            if diff == 0:
                diff = min_diff
        else:
            diff = spread

        old_delta = parent.delta
        if sr:
            lb = max(mu - diff, init_sr)
            ub = min(mu + diff, MOCKGenotype.sr_upper_bound)

            new_sr = int(random.uniform(lb, ub))
            new_delta = MOCKGenotype.sr_vals[new_sr]

        else:
            lb = max(mu - diff, init_delta)
            ub = min(mu + diff, max_delta)
            new_sr = None
            new_delta = round(random.uniform(lb, ub), precision)

        # Mutate delta
        parent.sr = new_sr
        parent.delta = new_delta

        # Update its genotype
        parent = MOCKGenotype.expand_reduce_genotype(parent, old_delta)

    return parent


def random_delta(parent, min_sr, min_delta, max_delta, precision, init_sr=None, init_delta=None, gen_n=None):
    sr = min_sr is not None
    old_delta = parent.delta

    # Change the delta of the parent to a new random number
    if sr:
        new_sr = randint(1, min_sr)
        new_delta = MOCKGenotype.sr_vals[new_sr]
        parent.sr = new_sr
        parent.delta = new_delta
    else:
        new_delta = randint(min_delta*10**precision, max_delta*10**precision+precision)
        parent.delta = new_delta/10**precision

    # Update its genotype
    parent = MOCKGenotype.expand_reduce_genotype(parent, old_delta)

    return parent


def min_max(value, mini, maxi, prec):
    return round(max(min(value, maxi), mini), prec)