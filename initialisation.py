import numpy as np
import random
import precompute

def baseGenotype(mst_genotype, int_links_indices, relev_links_len):
	base_genotype = mst_genotype.copy() # should use [:] for consistency
	# Remove the most interesting links to create the base components
	for index in int_links_indices[:relev_links_len]:
		base_genotype[index] = index
	base_clusters = precompute.decodingLAR(base_genotype)
	return base_genotype, base_clusters

def relevantLinks(delta_val, num_examples):
	# Length of the relevant links (symbol: capital gamma)
	return int(np.ceil(((100-delta_val)/100)*num_examples))

# def unfixedInterestLinks(int_links_indices, fixed_links):
# 	int_links_valid = int_links_indices.copy()
# 	for index in int_links_indices:
# 		# If this interesting link is in the fixed set, remove it from list
# 		if fixed_links[index] == True:
# 			# print("Link with index",index,"is in the fixed set")
# 			int_links_valid.remove(index)
# 	return int_links_valid

def replaceLink(argsortdists, i, j, L):
	# Link can be replaced with L+1 options
	# L nearest neighbours and self-connecting link
	# Must exclude replacing with original link
	while True:
		# L+1 accounts for self-connecting link and L nearest neighbours
		new_j = random.choice(argsortdists[i][0:L+1])
		# Only break if we have made a new connection, otherwise try again
		if new_j != j:
			break
	return new_j

# def fixedLinks(mst_genotype, int_links_indices, relev_links_len):
# 	fixed_links = [True] * len(mst_genotype)
# 	for index in int_links_indices[:relev_links_len]:
# 		fixed_links[index] = False
# 	return fixed_links

def initCreateSol(n, mst_genotype, int_links_indices, relev_links_len, argsortdists, L):
	indiv = mst_genotype[:]
	# print("Outside while loop! indiv has been reset")
	while True:
		# Need to only choose :n (k-1) from the unfixed set
		for index in int_links_indices[:relev_links_len][:n]:
			# print(n, len(int_links_indices[:relev_links_len][:n]))
			j = indiv[index]
			indiv[index] = replaceLink(argsortdists, index, j, L)
			# print("orig:",j,"new:",indiv[index])
		yield indiv

def initDeltaMOCK(k_user, num_indivs, mst_genotype, int_links_indices, relev_links_len, argsortdists, L):
	pop = []

	# Add MST to population
	indiv = mst_genotype[:]
	pop.append([indiv[i] for i in int_links_indices[:relev_links_len]]) # Add the MST solution

	# int_links_indices should be the list of indices/nodes where we have unfixed links
	k_max = k_user*2

	# Generate set of k values to use
	k_all = list(range(2,k_max+1))

	# Shuffle this set
	k_set = k_all[:]
	random.shuffle(k_set)
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

	for k in k_values:
		# n = k-1, with n being used in the Garza/Handl paper
		indiv = next(initCreateSol(k-1, mst_genotype, int_links_indices, relev_links_len, argsortdists, L))
		red_genotype = [indiv[i] for i in int_links_indices[:relev_links_len]]
		pop.append(red_genotype)
	return pop

# This function is used in the reinitialisation strategies, as we don't want to add the MST back in
def initDeltaMOCKadapt(k_user, num_indivs, mst_genotype, int_links_indices, relev_links_len, argsortdists, L):
	pop = []

	# int_links_indices should be the list of indices/nodes where we have unfixed links
	k_max = k_user*2

	# Generate set of k values to use
	k_all = list(range(2,k_max+1))

	# Shuffle this set
	k_set = k_all[:]
	random.shuffle(k_set)
	k_values = []

	# If we don't have enough k values we need to resample
	if len(k_all) < num_indivs:
		while len(k_values) < num_indivs:
			try:
				k_values.append(k_set.pop())

			except IndexError:
				k_set = k_all[:]
				random.shuffle(k_set)

	# Otherwise just take the number of k values we need from the shuffled set
	else:
		k_values = k_set[:num_indivs]

	assert len(k_values) == num_indivs, "Different number of k values to P (pop size)"

	for k in k_values:
		# n = k-1
		indiv = next(initCreateSol(k-1, mst_genotype, int_links_indices, relev_links_len, argsortdists, L))
		red_genotype = [indiv[i] for i in int_links_indices[:relev_links_len]]
		pop.append(red_genotype)
	return pop