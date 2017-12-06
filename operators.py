import random
import numpy as np
from deap import base
from initialisation import replaceLink
# from numba import jit

# @profile
def uniformCrossover(parent1, parent2, cxpb):
	# Below not really necessary
	assert len(parent1) == len(parent2)

	child1 = parent1[:]
	child2 = parent2[:]

	if cxpb == 1:
		for i in range(len(parent1)):
			if random.random() < 0.5:
				parent1[i] = child1[i]
				parent2[i] = child2[i]
			else:
				parent1[i] = child2[i]
				parent2[i] = child1[i]

		# Caveat: there is a 0.5**len(parent1) chance the parents will be equal to children
		del parent1.fitness.values, parent2.fitness.values

	# In case another probability is used, we avoid a random.random() call in normal case
	elif random.random() <= cxpb:
		for i in range(len(parent1)):
			if random.random() < 0.5:
				parent1[i] = child1[i]
				parent2[i] = child2[i]
			else:
				parent1[i] = child2[i]
				parent2[i] = child1[i]
		del parent1.fitness.values, parent2.fitness.values

	# If we change cxpb to be <1 then we may not enter loop, so return unchanged
	# We'll keep their fitnesses so we don't need to re-evaluate
	return parent1, parent2

# @profile
def neighbourMutation(parent, MUTPB, gen_length, argsortdists, L, int_links_indices, nn_rankings):
	first_term = (MUTPB / gen_length)

	# Using a comprehension for this bit is faster
	mutprobs = [first_term + ((nn_rankings[int_links_indices[index]][value] / gen_length) ** 2) for index,value in enumerate(parent)]

	# Now just loop over the probabilities
	# As we're using assignment, can't really do this part in a comprehension!
	for index, mutprob in enumerate(mutprobs):
		if random.random() < mutprob:
			# a,b = int_links_indices[index],parent[index]
			parent[index] = replaceLink(argsortdists, int_links_indices[index], parent[index], L)

	return parent

def neighbourMutationAdapt(parent, MUTPB, gen_length, argsortdists, L, int_links_indices, nn_rankings, old_length):
	first_term = (MUTPB / gen_length)

	# Using a comprehension for this bit is faster
	mutprobs = [first_term + ((nn_rankings[int_links_indices[index]][value] / gen_length) ** 2) for index,value in enumerate(parent[:old_length])]
	mutprobs.extend([(first_term + ((nn_rankings[int_links_indices[index+old_length]][value] / gen_length) ** 2))*100 for index,value in enumerate(parent[old_length:])])

	# print(mutprobs[:5], mutprobs[-5:])

	if len(parent) != len(mutprobs):
		raise ValueError("Insufficient mutation probabilities for genotype")

	# Now just loop over the probabilities
	# As we're using assignment, can't really do this part in a comprehension!
	for index, mutprob in enumerate(mutprobs):
		if random.random() < mutprob:
			# a,b = int_links_indices[index],parent[index]
			parent[index] = replaceLink(argsortdists, int_links_indices[index], parent[index], L)

	return parent

def neighbourHyperMutation_all(parent, MUTPB, gen_length, argsortdists, L, int_links_indices, nn_rankings, hyper_mut):
	first_term = (MUTPB / gen_length)

	# Using a comprehension for this bit is faster
	mutprobs = [(first_term + ((nn_rankings[int_links_indices[index]][value] / gen_length) ** 2))*hyper_mut for index,value in enumerate(parent)]

	# print(mutprobs,"\n")

	# Now just loop over the probabilities
	# As we're using assignment, can't really do this part in a comprehension!
	for index, mutprob in enumerate(mutprobs):
		if random.random() < mutprob:
			# a,b = int_links_indices[index],parent[index]
			parent[index] = replaceLink(argsortdists, int_links_indices[index], parent[index], L)

	return parent

def neighbourHyperMutation_spec(parent, MUTPB, gen_length, argsortdists, L, int_links_indices, nn_rankings, hyper_mut, new_genes):
	first_term = (MUTPB / gen_length)

	### Need to fix this to apply the high rate to only the new genes
	# See notebook

	# Using a comprehension for this bit is faster
	mutprobs = [first_term + ((nn_rankings[int_links_indices[index]][value] / gen_length) ** 2) for index,value in enumerate(parent)]

	# Now multiply the probabilities for the newly introduced variables
	new_probs = [mut_value * hyper_mut for mut_value in mutprobs[-len(new_genes):]]
	mutprobs[-len(new_genes):] = new_probs

	# Now just loop over the probabilities
	# As we're using assignment, can't really do this part in a comprehension!
	for index, mutprob in enumerate(mutprobs):
		if random.random() < mutprob:
			# a,b = int_links_indices[index],parent[index]
			parent[index] = replaceLink(argsortdists, int_links_indices[index], parent[index], L)

	return parent

def neighbourFairMutation(parent, MUTPB, gen_length, argsortdists, L, int_links_indices, nn_rankings, raised_mut):
	first_term = (MUTPB / gen_length)

	# Using a comprehension for this bit is faster
	mutprobs = [(first_term + ((nn_rankings[int_links_indices[index]][value] / gen_length) ** 2))*raised_mut for index,value in enumerate(parent)]

	# print(mutprobs,"\n")

	# Now just loop over the probabilities
	# As we're using assignment, can't really do this part in a comprehension!
	for index, mutprob in enumerate(mutprobs):
		if random.random() < mutprob:
			# a,b = int_links_indices[index],parent[index]
			parent[index] = replaceLink(argsortdists, int_links_indices[index], parent[index], L)

	return parent