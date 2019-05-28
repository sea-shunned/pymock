import random

from classes import MOCKGenotype

def create_solution(n, argsortdists, L):
    """Creates a single solution
    
    Arguments:
        n {int} -- Number of top-ranking links to remove
        argsortdists {np.array} -- Argsorted distance array
        L {int} -- Neighbourhood hyperparameter
    """
    # Start with the MST
    indiv = MOCKGenotype.mst_genotype[:]
    while True:
        # Need to only choose :n (k-1) from the unfixed set
        for index in MOCKGenotype.reduced_genotype_indices[:n]:
            # Get value at the gene
            j = indiv[index]
            # Replace the link
            indiv[index] = MOCKGenotype.replace_link(argsortdists, index, j, L)
        yield indiv

def init_deltamock(k_user, num_indivs, argsortdists, L):
    """Initialisation routine for Delta-MOCK
    
    Arguments:
        k_user {int} -- Rough estimate of the number of clusters
        num_indivs {int} -- Number of individuals in population
        argsortdists {np.array} -- Argsorted distance array
        L {int} -- Neighbourhood hyperparameter
    
    Returns:
        pop {list} -- Initial population
    """
    # Create empty list for population
    pop = []
    # Add MST to population
    indiv = MOCKGenotype.mst_genotype[:]
    pop.append([indiv[i] for i in MOCKGenotype.reduced_genotype_indices])
    # Set k_max to be 2* the k_user
    k_max = k_user*2
    # Generate set of k values to use
    k_all = list(range(2, k_max+1))
    # Shuffle this set
    k_set = k_all[:]
    random.shuffle(k_set)
    # Empty list to hold selected k values
    k_values = []
    # If we don't have enough k values we need to resample
    if len(k_all) < num_indivs-1:
        while len(k_values) < num_indivs-1:
            try:
                k_values.append(k_set.pop())
            except IndexError:
                k_set = k_all[:]
                random.shuffle(k_set)
    # Otherwise just take the number of k values we need from the shuffled set
    else:
        k_values = k_set[:num_indivs-1] # Minus one due to use of MST genotype
    assert len(k_values) == num_indivs-1, "Different number of k values to P (pop size)"
    # Generate each individual
    for k in k_values:
        # n = k-1, with n being used in the Garza/Handl paper
        indiv = next(create_solution(k-1, argsortdists, L))
        red_genotype = [indiv[i] for i in MOCKGenotype.reduced_genotype_indices]
        pop.append(red_genotype)
    return pop
