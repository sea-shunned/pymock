import numpy as np
from classes import MOCKGenotype
import igraph

MOCKGenotype.mst_genotype = [0, 4, 1, 1, 0, 2, 5, 6, 6, 2, 9, 9]
MOCKGenotype.degree_int = [0, 4.4, 6.1, 1.3, 1.1, 3.2, 5.5, 1.4, 2.3, 4.8, 1.7, 1.2]

# MOCKGenotype.calculate_delta(1)
# print(MOCKGenotype.delta_val)

MOCKGenotype.delta_val = 80

MOCKGenotype.interest_indices = np.argsort(-(np.asarray(MOCKGenotype.degree_int)), kind='mergesort').tolist()

MOCKGenotype.reduced_length = int(np.ceil(((100-MOCKGenotype.delta_val)/100)*len(MOCKGenotype.mst_genotype)))
print(MOCKGenotype.reduced_length)

MOCKGenotype.reduced_genotype_indices = MOCKGenotype.interest_indices[:MOCKGenotype.reduced_length]
print(MOCKGenotype.reduced_genotype_indices)

# print(MOCKGenotype.interest_indices)

indiv = MOCKGenotype()

# print(indiv.full_genotype)

indiv.reduce_genotype()
print("")
print(MOCKGenotype.mst_genotype)
print(indiv.genotype,"\n")

indiv.expand_genotype()
print(indiv.full_genotype)

# indiv.genotype[0] = 11
# indiv.expand_genotype()
# print(indiv.genotype)
# print(indiv.full_genotype)

MOCKGenotype.base_genotype()
print(MOCKGenotype.base_genotype)

g = igraph.Graph()
g.add_vertices(len(MOCKGenotype.base_genotype))
g.add_edges(zip(
    range(len(MOCKGenotype.base_genotype)),
    MOCKGenotype.base_genotype))

print(list(g.components(mode="WEAK")))

MOCKGenotype.base_clusters()
print(MOCKGenotype.base_clusters)


indiv.genotype = [3,8,5]
indiv.decode_genotype()
print(indiv.full_genotype)