import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
import os
from itertools import cycle


plt.style.use('seaborn-paper')

def normEAF(folder_path):
	files = glob.glob(folder_path+'*.csv*')

	# print(files)

	# Get the unique dataset names
	# uniq_data = set(['_'.join(file.split("/")[-1].split("_")[:-4]) for file in files])

	# Set to 0 or infinity so that we ensure we update
	max_val_var = 0
	min_val_var = np.inf

	max_val_cnn = 0
	min_val_cnn = np.inf # always 0 due to MST solution, check with Julia and remove

	for file in files:
		data = np.genfromtxt(file, delimiter=" ")
		data = data[:,:2]
		# print(data)
		
		curr_max_val_var = np.max(data[:,0])
		curr_min_val_var = np.min(data[:,0])

		curr_max_val_cnn = np.max(data[:,1])
		curr_min_val_cnn = np.min(data[:,1])

		# Remember signs!
		if max_val_var < curr_max_val_var:
			max_val_var = curr_max_val_var

		if min_val_var > curr_min_val_var:
			min_val_var = curr_min_val_var

		if max_val_cnn < curr_max_val_cnn:
			max_val_cnn = curr_max_val_cnn

		if min_val_cnn > curr_min_val_cnn:
			min_val_cnn = curr_min_val_cnn

	# print(max_val_var, max_val_cnn)
	# print(min_val_var, min_val_cnn)

	denom_var = max_val_var - min_val_var
	denom_cnn = max_val_cnn - min_val_cnn

	for file in files:
		# Normalise the data
		data = np.genfromtxt(file, delimiter=" ")
		data[:,0] = (data[:,0] - min_val_var)/denom_var
		data[:,1] = (data[:,1] - min_val_cnn)/denom_cnn

		# print(data)

		name = file.split(".")[0]+"_norm.csv"
		# print(file.split(".")[0])
		print(name)
		np.savetxt(name, data, delimiter = " ")
		# np.savetxt(results_basepath+classes.Dataset.data_name+"_eaf_"+str(delta)+".csv", arr, delimiter=" ")

	### From precompute normdist func
	# max_val = np.max(data)
	# min_val = np.min(data)
	# denom = max_val - min_val
	# for row, val in enumerate(distarray):
	# 	distarray[row] = (val - min_val)/denom


def plotObjectives(csv_path):
	data = np.genfromtxt(csv_path, delimiter=" ")

	if data.shape[1] == 3:
		data = np.delete(data, -1, 1)

	fig = plt.figure()
	ax = fig.add_subplot(111)
	plt.scatter(data[0:99,0], data[0:99,1], s=2, marker="x")
	plt.show()

# https://matplotlib.org/examples/color/color_cycle_demo.html
# def plotHV(csv_path):
# 	df_hv = pd.read_csv(csv_path)

# 	fig = plt.figure()
# 	ax = fig.add_subplot(111)

# 	# is below a list or a generator?
# 	colors = plt.cm.rainbow(np.linspace(0, 5, len(df_hv)))
# 	# colors = iter(cm.rainbow(np.linspace(0, 1, len(df_hv))))

# 	for index, column in enumerate(df_hv):
# 		# print(len(df_hv[column]))
# 		plt.plot(range(0,100), df_hv[column], color=colors[index], label=list(df_hv)[index]) #next(colors)
# 	plt.legend()
# 	plt.show()

# def plotHV_adaptdelta(HV, adapt_gens):
# 	fig = plt.figure()
# 	ax = fig.add_subplot(111)

# 	# print(HV)
# 	# print(adapt_gens)

# 	# This is to create some nicer max/min limits for the y-xis (HV)
# 	max_HV = round(np.ceil(np.max(HV)), -1)
# 	min_HV = round(np.floor(np.min(HV)), -1)

# 	ax.plot(range(0, len(HV)), HV, 'g-')
# 	for gen in adapt_gens:
# 		ax.plot([gen,gen], [0,max_HV+10], 'r--' )

# 	# -10 and +10 to avoid issues with rounding, to give some distance for our max and min
# 	ax.set_ylim([min_HV-10,max_HV+10])

# 	plt.show()

# 	# return ax

# def plotHVgens(folder_path, delta, styles_cycler, graph_path):
# 	files = glob.glob(folder_path+os.sep+"*"+"HVgens"+"*")
# 	print(folder_path)


# 	for file in files:

# 		# Read the csv in, and filter just the columns with the delta value we're plotting
# 		df = pd.read_csv(file)

# 		# Account for differences with numerical and sr5 etc.
# 		if isinstance(delta,int):
# 			df = df.filter(regex="d"+str(delta))
# 		else:
# 			df = df.filter(regex=delta)

# 		num_gens = df.shape[0]

# 		fig = plt.figure(figsize=(18,12))
# 		ax = fig.add_subplot(111)

# 		for i in range(0,len(df.columns),3):
# 			strat_name = df.columns[i].split("_")[2]

# 			ax.errorbar(list(range(0,num_gens)),df[df.columns[i]],
# 				yerr=df[df.columns[i+2]],
# 				label=strat_name,
# 				**next(styles_cycler)
# 				)

# 			ax.set_title("HV during Evolution for "+folder_path.split(os.sep)[-2])
# 			ax.set_xlabel("Generation")
# 			ax.set_ylabel("Hypervolume")
# 			ax.legend()

# 		# print(fig.dpi)
# 		plt.show()
# 		# print(fig.dpi)

# 		savename = graph_path+file.split('/')[-1].split('-')[0]+'-d'+str(delta)+'-HVplot.svg'
# 		# fig.savefig(savename, format='svg', dpi=1200, bbox_inches='tight')

def plotHVgens(folder_path, delta, graph_path, styles, save=False):
	files = glob.glob(folder_path+os.sep+"*"+"HV-"+str(delta)+"*")
	files.sort()

	if len(files) == 0:
		return

	data_name = folder_path.split(os.sep)[-1]

	styles_cycler = cycle(styles)

	ax, fig = graphHVgens(files, data_name, styles_cycler)
	
	if save:
		if isinstance(delta,int):
			savename = graph_path+os.sep+data_name+'-d'+str(delta)+'-ARIboxplot.svg'
		else:
			savename = graph_path+os.sep+data_name+'-'+delta+'-ARIboxplot.svg'
		
		fig.savefig(savename, format='svg', dpi=1200, bbox_inches='tight')		

	else:
		plt.show()

def graphHVgens(files, data_name, styles_cycler):
	fig = plt.figure(figsize=(18,12))
	ax = fig.add_subplot(111)

	for file in files:
		data = np.loadtxt(file, delimiter=',')

		# strat_name = file.split(os.sep)[-1].split("-")[1].split("_")[-1]
		strat_delta = "-".join([file.split(os.sep)[-1].split("-")[1].split("_")[-1],
			file.split(os.sep)[-1].split("-")[-1].split(".")[0]])

		ax.errorbar(list(range(0,data.shape[0])), np.mean(data,axis=1),
			yerr=np.std(data, ddof=0, axis=1)/np.sqrt(data.shape[1]),
			label=strat_delta,
			**next(styles_cycler))

	ax.set_xlabel("Generations")
	ax.set_ylabel("Hypervolume")
	ax.legend()

	ax.set_title("HV for {}".format(data_name))

	return ax, fig

def plotARI(folder_path, delta, graph_path, save=False):
	files = glob.glob(folder_path+os.sep+"*"+"ari-"+str(delta)+"*")
	files.sort()

	if len(files) == 0:
		return

	data_name = folder_path.split(os.sep)[-1]

	ax, fig = graphARI(files, data_name)

	if save:
		if isinstance(delta,int):
			savename = graph_path+os.sep+data_name+'-d'+str(delta)+'-ARIBoxplot.svg'
		else:
			savename = graph_path+os.sep+data_name+'-'+delta+'-ARIBoxplot.svg'

		fig.savefig(savename, format='svg', dpi=1200, bbox_inches='tight')

	else:
		plt.show()

def graphARI(files, data_name):
	fig = plt.figure(figsize=(18,12))
	ax = fig.add_subplot(111)

	data_list = []
	strat_names = []

	for file in files:
		data = np.loadtxt(file, delimiter=',')
		data_list.append(data)

		strat_delta = "-".join([file.split(os.sep)[-1].split("-")[1].split("_")[-1],
			file.split(os.sep)[-1].split("-")[-1].split(".")[0]])

		strat_names.append(strat_delta)

	ax.boxplot(data_list, labels=strat_names)
	ax.set_ylim(-0.05,1.05)

	ax.set_xlabel("Strategy")
	ax.set_ylabel("Adjusted Rand Index (ARI)")

	ax.set_title("ARI for "+data_name)

	return ax, fig

def plotNumClusts(folder_path, delta, graph_path, save=False):
	files = glob.glob(folder_path+os.sep+"*"+"numclusts-"+str(delta)+"*")
	files.sort()

	if len(files) == 0:
		return

	data_name = folder_path.split(os.sep)[-1]

	ax, fig = graphNumClusts(files, data_name)

	if save:
		if isinstance(delta,int):
			savename = graph_path+data_name+'-d'+str(delta)+'-NumClustsBoxplot.svg'
		else:
			savename = graph_path+os.sep+data_name+'-'+delta+'-NumClustsBoxplot.svg'

		fig.savefig(savename, format='svg', dpi=1200, bbox_inches='tight')

	else:
		plt.show()

def graphNumClusts(files, data_name):
	fig = plt.figure(figsize=(18,12))
	ax = fig.add_subplot(111)

	data_list = []
	strat_names = []

	for file in files:
		data = np.loadtxt(file, delimiter=',')
		data_list.append(data)

		strat_delta = "-".join([file.split(os.sep)[-1].split("-")[1].split("_")[-1],
			file.split(os.sep)[-1].split("-")[-1].split(".")[0]])

		strat_names.append(strat_delta)

	ax.boxplot(data_list, labels=strat_names)
	# ax.set_ylim(-0.05,1.05)

	ax.set_xlabel("Strategy")
	ax.set_ylabel("Number of Clusters")

	ax.set_title("Number of Clusters for {}".format(data_name))

	true_clusts = data_name.split("_")[-2]

	try:
		ax.plot(list(range(0,len(strat_names)+2)), [int(true_clusts)]*(len(strat_names)+2), linestyle = "--", label="True no. clusters")
	except ValueError:
		print("No true cluster number available")

	ax.legend()

	return ax, fig

def plotTimes(folder_path, delta, graph_path, styles_cycler, save=False):
	files = glob.glob(folder_path+os.sep+"*"+"time-"+str(delta)+"*")
	files.sort()

	if len(files) == 0:
		return

	fig = plt.figure(figsize=(18,12))
	ax = fig.add_subplot(111)

	data_name = folder_path.split(os.sep)[-1]

	width = 0.35

	ax, fig = graphTime(files, styles_cycler, data_name)

	if save:
		if isinstance(delta,int):
			savename = graph_path+data_name+'-d'+str(delta)+'-TimesPlot.svg'
		else:
			savename = graph_path+os.sep+data_name+'-'+delta+'-TimesPlot.svg'
		fig.savefig(savename, format='svg', dpi=1200, bbox_inches='tight')

	else:
		plt.show()

def graphTime(files, styles_cycler, data_name):
	fig = plt.figure(figsize=(18,12))
	ax = fig.add_subplot(111)

	strat_names = []
	width = 0.35

	for ind, file in enumerate(files):
		data = np.loadtxt(file, delimiter=',')
	
		strat_delta = "-".join([file.split(os.sep)[-1].split("-")[1].split("_")[-1],
			file.split(os.sep)[-1].split("-")[-1].split(".")[0]])

		strat_names.append(strat_delta)

		style = next(styles_cycler)
		
		# Use shape[0] as we just have a vector of length num_runs
		ax.bar(ind, np.mean(data), width, color=style['color'],
			yerr=np.std(data, ddof=0)/np.sqrt(data.shape[0]))

	ax.set_xticks(np.arange(len(strat_names)))
	ax.set_xticklabels((strat_names))

	ax.set_xlabel("Strategy")
	ax.set_ylabel("Time Taken")

	ax.set_title("Time taken for {}".format(data_name))

	return ax, fig

def fairmutComp(folder_path, delta, styles, save=False):
	files = glob.glob(folder_path+os.sep+"*"+"fairmut*"+str(delta)+"d*")

	# Skips folders where we haven't changed the delta_h value
	if len(files) == 0:
		return

	data_name = folder_path.split(os.sep)[-1]
	metrics = ["hv", "ari", "numclusts"]

	styles_cycler = cycle(styles)

	for metric in metrics:
		files = glob.glob(folder_path+os.sep+"*"+"fairmut*"+metric+"*"+str(delta)+"*")

		## Fix what files we send through etc.

		print(files)

		if metric == "hv":
			ax, fig = graphHVgens(files, data_name, styles_cycler)

		elif metric == "ari":
			ax, fig = graphARI(files, data_name)

		elif metric == "numclusts":
			ax, fig = graphNumClusts(files, data_name)

		plt.show()

def plotDeltaAssump(assumption_folder, graph_path, metric="ari", save=False):
	files = glob.glob(assumption_folder+os.sep+"*"+metric+"*")

	data_names = set()

	for file in files:
		data_names.add(file.split(os.sep)[-1].split("-")[0])

	print(data_names)

	fig = plt.figure(figsize=(18,12))
	ax = fig.add_subplot(111)

	for data_name in data_names:
		files = glob.glob(assumption_folder+os.sep+data_name+"*"+metric+"*")		
		# print(files,"\n")

		means = []
		stderrs = []
		delta_vals = []

		for file in files:
			data = np.loadtxt(file, delimiter=',')

			means.append(np.mean(data))
			stderrs.append(np.std(data, ddof=0)/np.sqrt(data.shape[1]))

			delta_vals.append(float(".".join(file.split(os.sep)[-1].split("-")[-1].split(".")[:-1])))

		
		ax.errorbar(delta_vals, means, yerr=stderrs, label=data_name, capsize=5, capthick=1)

	ax.set_xlabel("Delta Value")
	ax.set_ylabel(metric)

	ax.legend()

	if save:
		savename = graph_path+'DeltaAssumption.svg'
		print(savename)
		# fig.savefig(savename, format='svg', dpi=1200, bbox_inches='tight')

	else:
		plt.show()		


if __name__ == '__main__':
	basepath = os.getcwd()
	results_path = os.path.join(basepath, "results")
	# aggregate_folder = os.path.join(results_path, "aggregates")
	graph_path = os.path.join(results_path, "graphs")
	assumption_folder = os.path.join(results_path, "delta_assump")

	styles = [
	{'color':'b', 'dashes':(None,None), 'marker':"None"}, 		# base
	{'color':'r', 'dashes':(5,2), 'marker':"None",},			# carryon
	{'color':'g', 'dashes':(2,5), 'marker':"None",},			# hypermutspec
	{'color':'c', 'dashes':(None,None), 'marker':"o",'ms':7},	# hypermutall
	{'color':'m', 'dashes':(None,None), 'marker':"^",'ms':7},	# reinit
	{'color':'y', 'dashes':(None,None), 'marker':"D",'ms':7}]	# fairmut

	styles_cycler = cycle(styles)

	dataset_folders = glob.glob(results_path+os.sep+"*")
	# dataset_folders.remove(aggregate_folder)
	dataset_folders.remove(graph_path)

	delta = "sr5"

	save = False

	# for dataset in dataset_folders:
	# 	plotHVgens(dataset, delta, graph_path, styles, save)
		# plotARI(dataset, delta, graph_path, save)
		# plotNumClusts(dataset, delta, graph_path, save)
		# plotTimes(dataset, delta, graph_path, styles_cycler, save)

		# fairmutComp(dataset, delta, styles, save)



	# files = glob.glob(assumption_folder+os.sep+"*")

	plotDeltaAssump(assumption_folder, graph_path)