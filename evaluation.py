from sklearn.metrics import adjusted_rand_score
import classes
import igraph
import numpy as np
import pandas as pd


def finalPopMetrics2(pop, mst_genotype, int_links_indices, relev_links_len):
	final_pop_metrics = pd.DataFrame()

	num_examples = classes.Dataset.num_examples

	sol_num_clusters = []
	ari_values = []
	VARs = []
	CNNs = []

	for indiv in pop:
		# Create a graph
		g = igraph.Graph()

		# Add each point to the graph
		g.add_vertices(num_examples) # or len(mst_genotype))

		# Get the base nodes
		base_nodes = list(range(num_examples))

		# Get the mst_genotype
		new_values = np.asarray(mst_genotype)

		# Modify the relevant part of the mst_genotype
		new_values[int_links_indices[:relev_links_len]] = indiv

		# Create the graph
		g.add_edges(zip(base_nodes, new_values))

		# Get the connected components
		conn_components = g.components(mode="WEAK")
		sol_num_clusters.append(len(conn_components))

		# Now use these components to get labels
		pred_labels = np.empty(num_examples)
		pred_labels.fill(np.nan) # fill with NaNs so we know it works
		
		for i, component in enumerate(conn_components):
			pred_labels[component] = i

		if np.any(np.isnan(pred_labels)) == True:
			print("Label missing when trying to calculate ARI")

		# Add the ARI value
		ari_values.append(adjusted_rand_score(labels_true=classes.Dataset.label_vals, labels_pred=pred_labels))

		VARs.append(indiv.fitness.values[0])
		CNNs.append(indiv.fitness.values[1])

	# Add to the dataframe
	final_pop_metrics['Num Clusters'] = sol_num_clusters
	final_pop_metrics['ARI'] = ari_values
	final_pop_metrics['VAR'] = VARs
	final_pop_metrics['CNN'] = CNNs

	#### Might be best to return all of this individually
	
	return final_pop_metrics

# @profile
def finalPopMetrics(pop, mst_genotype, int_links_indices, relev_links_len):

	num_examples = classes.Dataset.num_examples

	sol_num_clusters = []
	ari_values = []

	for indiv in pop:
		# Create a graph
		g = igraph.Graph()

		# Add each point to the graph
		g.add_vertices(num_examples) # or len(mst_genotype))

		# Get the base nodes
		base_nodes = list(range(num_examples))

		# Get the mst_genotype
		new_values = np.asarray(mst_genotype)

		# Modify the relevant part of the mst_genotype
		new_values[int_links_indices[:relev_links_len]] = indiv

		# Create the graph
		g.add_edges(zip(base_nodes, new_values))

		# Get the connected components
		conn_components = g.components(mode="WEAK")
		sol_num_clusters.append(len(conn_components))

		# Now use these components to get labels
		pred_labels = np.empty(num_examples)
		pred_labels.fill(np.nan) # fill with NaNs so we know it works
		
		for i, component in enumerate(conn_components):
			pred_labels[component] = i

		assert np.any(np.isnan(pred_labels)) != True

		# Add the ARI value
		ari_values.append(adjusted_rand_score(labels_true=classes.Dataset.label_vals, labels_pred=pred_labels))

	return sol_num_clusters, ari_values


def numClusters(pop, mst_genotype, int_links_indices, relev_links_len):
	num_examples = classes.Dataset.num_examples

	sol_num_clusters = []

	for indiv in pop:
		# Create a graph
		g = igraph.Graph()

		# Add each point to the graph
		g.add_vertices(num_examples) # or len(mst_genotype))

		# Get the base nodes
		base_nodes = list(range(num_examples))

		# Get the mst_genotype
		new_values = np.asarray(mst_genotype)

		# Modify the relevant part of the mst_genotype
		new_values[int_links_indices[:relev_links_len]] = indiv

		# Create the graph
		g.add_edges(zip(base_nodes, new_values))

		# Get the connected components
		conn_components = g.components(mode="WEAK")
		sol_num_clusters.append(len(conn_components))

	return np.asarray(sol_num_clusters)