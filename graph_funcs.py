import pandas as pd
import glob
import numpy as np
# import matplotlib
# matplotlib.use('PS')
import matplotlib.pyplot as plt
import os
from itertools import cycle


plt.style.use('seaborn-paper')
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')

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

# 		savename = graph_path+file.split('/')[-1].split('-')[0]+'-d'+str(delta)+'-HVplot.pdf'
# 		# fig.savefig(savename, format='pdf', dpi=1200, bbox_inches='tight')

def plotHVgens(folder_path, delta, graph_path, styles, save=False):
	files = glob.glob(folder_path+os.sep+"*"+"hv-"+str(delta)+"*")
	files.sort()

	print(files)

	if len(files) == 0:
		return

	data_name = folder_path.split(os.sep)[-1]

	styles_cycler = cycle(styles)

	ax, fig = graphHVgens(files, data_name, styles_cycler)
	
	if save:
		if isinstance(delta,int):
			savename = graph_path+os.sep+data_name+'-d'+str(delta)+'-HVgenplot.pdf'
		else:
			savename = graph_path+os.sep+data_name+'-'+delta+'-HVgenplot.pdf'
		
		fig.savefig(savename, format='pdf', dpi=1200, bbox_inches='tight')		

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
			savename = graph_path+os.sep+data_name+'-d'+str(delta)+'-ARIBoxplot.pdf'
		else:
			savename = graph_path+os.sep+data_name+'-'+delta+'-ARIBoxplot.pdf'

		fig.savefig(savename, format='pdf', dpi=1200, bbox_inches='tight')

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
			savename = graph_path+'-d'+str(delta)+'-NumClustsBoxplot.pdf'
		else:
			savename = graph_path+os.sep+'-'+delta+'-NumClustsBoxplot.pdf'

		fig.savefig(savename, format='pdf', dpi=1200, bbox_inches='tight')

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
			savename = graph_path+data_name+'-d'+str(delta)+'-TimesPlot.pdf'
		else:
			savename = graph_path+os.sep+data_name+'-'+delta+'-TimesPlot.pdf'
		fig.savefig(savename, format='pdf', dpi=1200, bbox_inches='tight')

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

def fairmutComp(folder_path, graph_path, delta, styles, save=False):
	files = glob.glob(folder_path+os.sep+"*"+"fairmut*"+str(delta)+"d*")

	# Skips folders where we haven't changed the delta_h value
	if len(files) == 0:
		return

	data_name = folder_path.split(os.sep)[-1]
	metrics = ["hv", "numclusts"]

	styles_cycler = cycle(styles)

	for metric in metrics:
		files = glob.glob(folder_path+os.sep+"*"+"fairmut*"+metric+"*"+str(delta)+"*")

		## Fix what files we send through etc.

		if metric == "hv":
			ax, fig = graphHVgens(files, data_name, styles_cycler)

		elif metric == "ari":
			ax, fig = graphARI(files, data_name)

		elif metric == "numclusts":
			ax, fig = graphNumClusts(files, data_name)

		if save:
			savename = graph_path+data_name+"-fairmut-"+metric+".pdf"
			fig.savefig(savename, format='pdf', dpi=1200, bbox_inches='tight')
		else:
			plt.show()

def plotDeltaAssump(assumption_folder, graph_path, metric="ari", save=True):
	files = glob.glob(assumption_folder+os.sep+"*"+metric+"*")

	data_names = set()

	for file in files:
		data_names.add(file.split(os.sep)[-1].split("-")[0])

	print(data_names)

	fig = plt.figure(figsize=(18,12))
	ax = fig.add_subplot(111)

	for data_name in data_names:
		files = glob.glob(assumption_folder+os.sep+data_name+"*"+metric+"*")		
		files = sorted(files)


		means = []
		stderrs = []
		delta_vals = []

		for file in files:
			delta = float(".".join(file.split(os.sep)[-1].split("-")[-1].split(".")[:-1]))
			if delta < 90:
				continue

			data = np.loadtxt(file, delimiter=',')

			means.append(np.mean(data))
			stderrs.append(np.std(data, ddof=0)/np.sqrt(data.shape[1]))

			delta_vals.append(delta)

		
		ax.errorbar(delta_vals, means, yerr=stderrs, label=data_name, capsize=5, capthick=1)

	ax.set_xlabel("Delta Value")
	ax.set_ylabel(metric)

	ax.legend(loc='lower left')

	if save:
		savename = graph_path+'DeltaAssumption.pdf'
		# print(savename)
		fig.savefig(savename, format='pdf', dpi=1200, bbox_inches='tight')

	else:
		plt.show()


def plotArtifExp_single(dataset_folder,graph_path,metric="ari",save=False):
	metric_files = sorted(glob.glob(dataset_folder+os.sep+"*"+metric+"*"), reverse=True)
	time_files = sorted(glob.glob(dataset_folder+os.sep+"*"+"time"+"*"), reverse=True)

	assert len(metric_files) == len(time_files)

	fig = plt.figure(figsize=(18,12))
	ax1 = fig.add_subplot(111)

	width = 0.35

	means=[]
	errs=[]

	colors=["w","w","g","c","m","y","k"]
	hatches = ["/" , "\\" , "|" , "-" , "+" , "x", "."]

	data_name = dataset_folder.split(os.sep)[-1]

	strat_names = []
	for index, file in enumerate(metric_files):
		
		data_metric = np.loadtxt(file, delimiter=",")

		if "main_base" in file:
			strat_names.append("-".join([file.split(os.sep)[-1].split("-")[1].split("_")[-1],
				file.split(os.sep)[-1].split("-")[-1].split(".")[0][:3]]))

		else:
			strat_names.append(file.split(os.sep)[-1].split("-")[1].split("_")[-1])

		err = np.std(data_metric, ddof=0)/np.sqrt(data_metric.shape[1])

		ax1.bar(index, np.mean(data_metric), width, tick_label=strat_names[-1], label=strat_names[-1], alpha=0.7, lw=1.2, color="grey", edgecolor="black", hatch=hatches[index], yerr=err, error_kw=dict(capsize=3, capthick=1, ecolor="black", alpha=0.7))

		data_time = np.loadtxt(time_files[index], delimiter=',')
		
		means.append(np.mean(data_time))
		errs.append(np.std(data_time, ddof=0)/np.sqrt(data_time.shape[0]))

	ax1.set_ylabel("ARI")
	ax1.set_xlabel("Strategy")
	ax1.set_ylim(-0.05,1.05)

	ax2 = ax1.twinx()

	ax2.errorbar(list(range(len(strat_names))), means, color="black", linestyle="--", yerr=errs, capsize=7, capthick=1, label="Time")

	ax2.set_ylabel("Time")

	# print(strat_names)
	ax2.set_xticks(np.arange(len(strat_names)))
	ax2.set_xticklabels(strat_names)
	ax2.set_title("Comparison of time and performance for "+data_name)

	lines, labels = ax1.get_legend_handles_labels()
	lines2, labels2 = ax2.get_legend_handles_labels()
	ax2.legend(lines + lines2, labels + labels2, loc=0)

	if save:
		savename = graph_path + "artif-interval-" + data_name + "-bar.pdf"
		fig.savefig(savename, format='pdf', dpi=1200, bbox_inches='tight')
	else:
		plt.show()

def plotArtifExp_multiple(artif_folder, metric="ari"):
	# Use glob to get all files (recursively)
	folders = glob.glob(artif_folder+os.sep+"*", recursive=True)
	print(folders)

	# I'll need to plot the line graph (time) multiple times, one for each dataset in the right x positions
	# Easier alternative might be to use subplots

	# Equivalent to the number of datasets
	num_subplots = len(glob.glob(artif_folder+os.sep+"*"))

	# plt.subplots(1, num_subplots, sharex='all', sharey='all')

	width = 0.4

	colors=["w","w","g","c","m","y","k"]
	hatches = ["/" , "\\" , "|" , "-" , "+" , "x", "."]

	fig = plt.figure(figsize=(18,12))
	# ax1 = fig.add_subplot(111)

	for subplot_num, dataset_folder in enumerate(folders):
		subplot_num += 1

		ax1 = fig.add_subplot(num_subplots, 1, subplot_num)
		# ax1 = fig.add_subplot(1, num_subplots, subplot_num)

		metric_files = sorted(glob.glob(dataset_folder+os.sep+"*"+metric+"*"), reverse=True)
		time_files = sorted(glob.glob(dataset_folder+os.sep+"*"+"time"+"*"), reverse=True)

		means = []
		errs = []
		strat_names = []

		data_name = dataset_folder.split(os.sep)[-1]

		for index, file in enumerate(metric_files):
			data_metric = np.loadtxt(file, delimiter=",")

			if "main_base" in file:
				strat_names.append("-".join([file.split(os.sep)[-1].split("-")[1].split("_")[-1],
					file.split(os.sep)[-1].split("-")[-1].split(".")[0][:3]]))

			else:
				strat_names.append(file.split(os.sep)[-1].split("-")[1].split("_")[-1])

			err = np.std(data_metric, ddof=0)/np.sqrt(data_metric.shape[1])

			ax1.bar(index, np.mean(data_metric), width, tick_label=strat_names[-1], label=strat_names[-1], alpha=0.7, lw=1.2, color="grey", edgecolor="black", hatch=hatches[index], yerr=err, error_kw=dict(capsize=3, capthick=1, ecolor="black", alpha=0.7))

			data_time = np.loadtxt(time_files[index], delimiter=',')

			# Need to normalise the times here
		

			means.append(np.mean(data_time))
			errs.append(np.std(data_time, ddof=0)/np.sqrt(data_time.shape[0]))

		ax1.set_ylabel("ARI")
		ax1.set_xlabel("Strategy")
		ax1.set_ylim(-0.05,1.05)

		ax2 = ax1.twinx()

		ax2.errorbar(list(range(len(strat_names))), means, color="black", linestyle="--", yerr=errs, capsize=7, capthick=1, label="Time")

		# https://stackoverflow.com/questions/12919230/how-to-share-secondary-y-axis-between-subplots-in-matplotlib
		## Use the above to get_shared_y_axes()

		ax2.set_ylabel("Time")

		# print(strat_names)
		ax2.set_xticks(np.arange(len(strat_names)))
		ax2.set_xticklabels(strat_names)
		ax2.set_title("Comparison of time and performance for "+data_name)

	plt.show()

def plotArtifExp_multiple2(artif_folder, metric="ari"):
	folders = glob.glob(artif_folder+os.sep+"*", recursive=True)
	num_subplots = len(glob.glob(artif_folder+os.sep+"*"))

	ind = np.arange(int(num_subplots/2)*-2, int(num_subplots/2))
	ind = ind*0.26 # space between bars in a group

	width = 0.2

	colors = ["w","w","g","c","m","y","k"]
	hatches = ["/" , "\\" , "|" , "-" , "+" , "x", "."]

	fig = plt.figure(figsize=(18,12))
	ax1 = fig.add_subplot(111)
	ax2 = ax1.twinx()

	for dataset_num, dataset_folder in enumerate(folders):
		metric_files = sorted(glob.glob(dataset_folder+os.sep+"*"+metric+"*"), reverse=True)
		time_files = sorted(glob.glob(dataset_folder+os.sep+"*"+"time"+"*"), reverse=True)

		means = []
		errs = []
		strat_names = []

		data_name = dataset_folder.split(os.sep)[-1]
		
		# for loop strategy
		for index, file in enumerate(metric_files):
			data_metric = np.loadtxt(file, delimiter=",")

			if "main_base" in file:
				strat_names.append("-".join([file.split(os.sep)[-1].split("-")[1].split("_")[-1],
					file.split(os.sep)[-1].split("-")[-1].split(".")[0][:3]]))

			else:
				strat_names.append(file.split(os.sep)[-1].split("-")[1].split("_")[-1])

			err = np.std(data_metric, ddof=0)/np.sqrt(data_metric.shape[1])

			# Plot the bars here
			# ax1.bar(2*i+ind[j], ens_prob_avg[i][j], width=w, alpha=0.6 , color=colours[0].rgb, label='geometric comb')
			# ax1.bar(index, np.mean(data_metric), width, tick_label=strat_names[-1], label=strat_names[-1], alpha=0.7, lw=1.2, color="grey", edgecolor="black", hatch=hatches[index], yerr=err, error_kw=dict(capsize=3, capthick=1, ecolor="black", alpha=0.7))
			ax1.bar(2*dataset_num+ind[index], np.mean(data_metric), width, alpha=0.7, lw=1.2, color="grey", edgecolor="black", hatch=hatches[index], yerr=err, error_kw=dict(capsize=3, capthick=1, ecolor="black", alpha=0.6))

			data_time = np.loadtxt(time_files[index], delimiter=',')

			# Need to normalise the times here
			max_val = np.max(data_time)
			min_val = np.min(data_time)
			denom = max_val - min_val

			print(data_time)
			data_time = (data_time - min_val)/denom
			print(data_time)

			means.append(np.mean(data_time))
			errs.append(np.std(data_time, ddof=0)/np.sqrt(data_time.shape[0]))

		# Plot the line here for each dataset
		# Similar method to above, basically mirroring the bars but with lines
		# ax2 = ax1.twinx()

		# print(means, len(means))
		# print([i+2*dataset_num for i in ind], len([i+2*dataset_num for i in ind[:len(means)]]))

		ax2.errorbar([i+2*dataset_num for i in ind[:len(means)]], means, color="red", linestyle="--", yerr=errs, capsize=7, capthick=1)
		# ax2.errorbar(list(range(len(strat_names))), means, color="black", linestyle="--", yerr=errs, capsize=7, capthick=1, label="Time")

	ax2.set_xticks(np.arange(num_subplots)*2+ind[num_subplots])

	plt.show()


def plotArtifExp_singlebox(dataset_folder,graph_path,metric="ari",save=False):
	metric_files = sorted(glob.glob(dataset_folder+os.sep+"*"+metric+"*"), reverse=True)
	time_files = sorted(glob.glob(dataset_folder+os.sep+"*"+"time"+"*"), reverse=True)

	assert len(metric_files) == len(time_files)

	fig = plt.figure(figsize=(18,12))
	ax1 = fig.add_subplot(111)

	width = 0.35

	means=[]
	errs=[]

	colors=["w","w","g","c","m","y","k"]
	hatches = ["/" , "\\" , "|" , "-" , "+" , "x", "."]

	data_name = dataset_folder.split(os.sep)[-1]

	strat_names = []
	data_metric_list = []

	min_val = np.inf
	max_val = 0

	for index, file in enumerate(metric_files):
		
		data_metric_list.append(np.max(np.loadtxt(file, delimiter=","),axis=0))

		if "main_base" in file:
			strat_names.append("-".join([file.split(os.sep)[-1].split("-")[1].split("_")[-1],
				file.split(os.sep)[-1].split("-")[-1].split(".")[0][:3]]))

		else:
			strat_names.append(file.split(os.sep)[-1].split("-")[1].split("_")[-1])

		data_time = np.loadtxt(time_files[index], delimiter=',')

		if np.max(data_time) > max_val:
			max_val = np.max(data_time)

		if np.min(data_time) < min_val:
			min_val = np.min(data_time)

		denom = max_val - min_val

	for index, file in enumerate(time_files):
		data_time = np.loadtxt(file, delimiter=',')

		data_time = (data_time - min_val)/denom

		means.append(np.mean(data_time))
		errs.append(np.std(data_time, ddof=0)/np.sqrt(data_time.shape[0]))

	ax1.boxplot(data_metric_list, labels=strat_names)

	ax1.set_ylabel("ARI")
	ax1.set_xlabel("Strategy")
	ax1.set_ylim(-0.05,1.05)

	ax2 = ax1.twinx()

	ax2.errorbar(list(range(1,len(data_metric_list)+1)), means, color="black", linestyle="--", yerr=errs, capsize=7, capthick=1, label="Time")

	ax2.set_ylabel("Time")

	ax2.set_xticklabels(strat_names)
	ax2.set_title("Comparison of time and performance for "+data_name)
	ax2.set_ylim(-0.05,1.05)

	lines, labels = ax1.get_legend_handles_labels()
	lines2, labels2 = ax2.get_legend_handles_labels()
	ax2.legend(lines + lines2, labels + labels2, loc=1)

	if save:
		savename = graph_path + "artif-interval-" + data_name + "-box.pdf"
		fig.savefig(savename, format='pdf', dpi=1200, bbox_inches='tight')
	else:
		plt.show()

def plotArtifExp_multiplebox(artif_folder, metric="ari"):
	folders = glob.glob(artif_folder+os.sep+"*", recursive=True)
	num_subplots = len(glob.glob(artif_folder+os.sep+"*"))

	ind = np.arange(int(num_subplots/2)*-2, int(num_subplots/2))
	ind = ind*0.26 # space between bars in a group

	width = 0.2

	colors = ["w","w","g","c","m","y","k"]
	hatches = ["/" , "\\" , "|" , "-" , "+" , "x", "."]

	fig = plt.figure(figsize=(18,12))
	# ax1 = fig.add_subplot(111)
	

	for subplot_num, dataset_folder in enumerate(folders):
		subplot_num += 1

		if subplot_num > 1:
			ax1 = fig.add_subplot(num_subplots, 2, subplot_num, sharex=fig.get_axes()[0])
		else:
			ax1 = fig.add_subplot(num_subplots, 2, subplot_num)
		# ax1 = fig.add_subplot(1, num_subplots, subplot_num)



		metric_files = sorted(glob.glob(dataset_folder+os.sep+"*"+metric+"*"), reverse=True)
		time_files = sorted(glob.glob(dataset_folder+os.sep+"*"+"time"+"*"), reverse=True)

		means = []
		errs = []
		strat_names = []
		data_metric_list = []

		data_name = dataset_folder.split(os.sep)[-1]

		min_val = np.inf
		max_val = 0
		
		# for loop strategy
		for index, file in enumerate(metric_files):

			data_metric_list.append(np.max(np.loadtxt(file, delimiter=","),axis=0))

			if "main_base" in file:
				strat_names.append("-".join([file.split(os.sep)[-1].split("-")[1].split("_")[-1],
					file.split(os.sep)[-1].split("-")[-1].split(".")[0][:3]]))

			else:
				strat_names.append(file.split(os.sep)[-1].split("-")[1].split("_")[-1])


			# ax1.bar(2*subplot_num+ind[index], np.mean(data_metric), width, alpha=0.7, lw=1.2, color="grey", edgecolor="black", hatch=hatches[index], yerr=err, error_kw=dict(capsize=3, capthick=1, ecolor="black", alpha=0.6))

			data_time = np.loadtxt(time_files[index], delimiter=',')

			# Need to normalise the times here
			##### Not here, we need to loop through and find the max over all strategies first!
			# max_val = np.max(data_time)
			# min_val = np.min(data_time)
			# denom = max_val - min_val

			if np.max(data_time) > max_val:
				max_val = np.max(data_time)

			if np.min(data_time) < min_val:
				min_val = np.min(data_time)


		denom = max_val - min_val

		for index, file in enumerate(time_files):
			data_time = np.loadtxt(file, delimiter=',')

			data_time = (data_time - min_val)/denom

			means.append(np.mean(data_time))
			errs.append(np.std(data_time, ddof=0)/np.sqrt(data_time.shape[0]))
			

		ax1.boxplot(data_metric_list, labels=strat_names)
		# ax1.set_ylabel("ARI")
		# ax1.set_xlabel("Strategy")
		ax1.set_ylim(-0.05,1.05)

		ax2 = ax1.twinx()

		ax2.errorbar(list(range(1,len(data_metric_list)+1)), means, color="red", linestyle="--", yerr=errs, capsize=7, capthick=1)
		# ax2.errorbar(list(range(len(strat_names))), means, color="black", linestyle="--", yerr=errs, capsize=7, capthick=1, label="Time")

		# ax2.set_ylabel("Time")
		ax2.set_ylim(-0.05,1.05)
		# ax2.set_xticks(np.arange(len(strat_names)))
		ax2.set_xticklabels(strat_names)

	ax1.set_ylabel("ARI")
	ax1.set_xlabel("Strategy")
	ax2.set_ylabel("Time")

	# print(len(fig.get_axes()))

	axes = fig.get_axes()
	# print(axes, axes[0])

	# for i in range(2, len(axes),2):
	# 	print(axes[i])
		# axes[0].get_shared_x_axes().join(axes[0],axes[i])
		# axes[0].get_shared_y_axes().join(axes[0],axes[i])

	# -4 just means the bottom 2 with 7 dataset results
	for i in range(len(axes)-4):
		plt.setp(axes[i].get_xticklabels(), visible=False)

	plt.show()


	# Consider saving these axes into a dict or something, and then adding them to a fig afterwards with a gridspec subplot - may be able to control axes better/make it look better


if __name__ == '__main__':
	basepath = os.getcwd()
	results_path = os.path.join(basepath, "results")
	# aggregate_folder = os.path.join(results_path, "aggregates")
	graph_path = os.path.join(results_path, "graphs")
	assumption_folder = os.path.join(results_path, "delta_assump")
	artif_folder = os.path.join(results_path, "artif")

	styles = [
	{'color':'b', 'dashes':(None,None), 'marker':"None"}, 		# base
	{'color':'r', 'dashes':(5,2), 'marker':"None",},			# carryon
	{'color':'g', 'dashes':(2,5), 'marker':"None",},			# fairmut
	{'color':'c', 'dashes':(None,None), 'marker':"o",'ms':7},	# fairmut extra
	{'color':'m', 'dashes':(None,None), 'marker':"^",'ms':7},	# hypermutall
	{'color':'y', 'dashes':(None,None), 'marker':"D",'ms':7},	# hypermutspec
	{'color':'k', 'dashes':(None,None), 'marker':"v",'ms':7}]	# reinit

	styles_cycler = cycle(styles)

	dataset_folders = glob.glob(results_path+os.sep+"*")
	# dataset_folders.remove(aggregate_folder)
	dataset_folders.remove(graph_path)
	dataset_folders.remove(artif_folder)

	graph_path = os.path.join(results_path, "graphs")+os.sep

	delta = "sr5"

	save = False

	# font = {'family' : 'normal',
	# 	'weight' : 'medium',
	# 	'size'   : 12}

	# plt.rc('font', **font)

	SMALL_SIZE = 12
	MEDIUM_SIZE = 14
	BIGGER_SIZE = 16
	plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
	# plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
	plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
	plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
	plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
	plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
	plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

	# for dataset in dataset_folders:

	# # 	plotHVgens(dataset, delta, graph_path, styles, save)
	# 	# plotARI(dataset, delta, graph_path, save)
	# 	# plotNumClusts(dataset, delta, graph_path, save)
	# 	# plotTimes(dataset, delta, graph_path, styles_cycler, save)


	# 	if "200_20_2" in dataset:
	# 		print(dataset)
	# 		# fairmutComp(dataset, graph_path, delta, styles, True)
	# 		plotNumClusts(dataset, delta, graph_path, False)
	# 		# plt.rc('text', usetex=True)
	# 		plt.rc('font', family='serif')
	# 		plotNumClusts(dataset, delta, graph_path, False)
	# 		plotHVgens(dataset, delta, graph_path, styles, False)
	# 	elif "20_100_10" in dataset:
	# 		print(dataset)
	# 		# fairmutComp(dataset, graph_path, delta, styles, True)
	# 		plotNumClusts(dataset, delta, graph_path, False)
	# 		plotHVgens(dataset, delta, graph_path, styles, False)

	# artif_folder = os.path.join(results_path, "artif")+os.sep
	dataset_folders = glob.glob(artif_folder+os.sep+"*")

	# for dataset_folder in dataset_folders:
	# 	plotArtifExp_singlebox(dataset_folder,graph_path,metric="ari",save=False)

	# plotArtifExp_multiple(artif_folder)
	# plotArtifExp_multiple2(artif_folder)
	plotArtifExp_multiplebox(artif_folder)