from deap import base, creator, tools
from operators import gaussian_mutation_delta
from classes import MOCKGenotype
import random
import warnings


def get_n_genes(delta, total):
    return int(round((100 - delta) / 100 * total, 0))


def get_random_delta(min_delta, max_delta, mst, precision=3):
    if min_delta > max_delta:
        warnings.warn('Swapping min and max delta...', Warning)
        min_delta, max_delta = max_delta, min_delta
    if max_delta > 100:
        warnings.warn('Setting max_delta to 100...', Warning)
        max_delta = 100
    if min_delta < 0:
        warnings.warn('Setting min_delta to 0...', Warning)
        max_delta = 0

    delta_diff = max_delta - min_delta
    delta = round(random.random() * delta_diff + min_delta, precision)
    n_genes = get_n_genes(delta, len(mst))

    return delta, n_genes


def init_individual(icls, min_delta, max_delta, mst, di_index, delta=None, precision=3):
    if delta is None:
        # Get the value for delta and the number of genes it represents
        delta, n_genes = get_random_delta(min_delta, max_delta, mst, precision)
    else:
        n_genes = get_n_genes(delta, len(mst))
        
    ind = icls([mst[i] for i in di_index[:n_genes]])
    ind.delta = delta
    return ind


def rebuild_ind_genotype(genotype, di_index, mst):
    """
    :param genotype: list. Reduced individual's genotype
    :param di: list. DI index
    :param mst: list. full mst
    :return: full individual's genotype
    """
    genotype_index = di_index[:len(genotype)]
    for i, idx in enumerate(genotype_index):
        mst[idx] = genotype[i]
    return mst


# Test function
def init_test(n=20, nclusters=3, min_delta=80, max_delta=99):
    full_genotype = [0]
    j = 1
    for i in range(1, n):
        if i <= n*j/nclusters:
            full_genotype.append(i-1)
        else:
            full_genotype.append(i)
            j += 1

    creator.create("Fitness", base.Fitness, weights=(-1.0, -1.0))
    creator.create("Individual", list, fitness=creator.Fitness, fairmut=None)

    toolbox = base.Toolbox()
    toolbox.register('individual', init_individual, creator.Individual, min_delta,
                     max_delta, full_genotype, full_genotype)
    return full_genotype, toolbox


if __name__ == '__main__':
    full_genotype, toolbox = init_test()
    ind = toolbox.individual(delta=90)
    print(full_genotype)
    print(ind)
    print(type(ind))
    print(ind.delta)

    ind[1] = 99
    print(ind)

    MOCKGenotype.n_links = len(full_genotype)
    MOCKGenotype.mst_genotype = full_genotype
    ind = gaussian_mutation_delta(ind, 10, 1, precision=0)
    print(ind)
    print(ind.delta)
    print(ind.fitness.valid)

