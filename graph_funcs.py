import glob
import os
import random
from itertools import cycle

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# import matplotlib
# matplotlib.use('PS')

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
        # np.savetxt(results_basepath+classes.Datapoint.data_name+"_eaf_"+str(delta)+".csv", arr, delimiter=" ")

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

    ax.set_xlabel("Search Strategy")
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

    ax.set_xlabel("Search Strategy")
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

    ax.set_xlabel("Search Strategy")
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

def plotDeltaAssump(assumption_folder, graph_path, styles_cycler, metric="ari", save=False):
    files = glob.glob(assumption_folder+os.sep+"*"+metric+"*")

    data_names = set()

    for file in files:
        data_names.add(file.split(os.sep)[-1].split("-")[0])
        #add if "_" in data_name to reformat tevc data names

    print(data_names)

    fig = plt.figure(figsize=(18,12))
    ax = fig.add_subplot(111)

    for data_name in data_names:
        files = glob.glob(assumption_folder+os.sep+data_name+"*"+metric+"*")		
        files = sorted(files)

        means = []
        stderrs = []
        delta_vals = []
        data_list = []

        for file in files:
            delta = float(".".join(file.split(os.sep)[-1].split("-")[-1].split(".")[:-1]))
            if delta < 90:
                continue

            data_list.append(np.max(np.loadtxt(file, delimiter=","),axis=0))

            delta_vals.append(delta)

            means.append(np.mean(data_list))
            stderrs.append(np.std(data_list, ddof=0)/np.sqrt(len(data_list)))

        ax.errorbar(delta_vals, means, yerr=stderrs, label=data_name, capsize=5, capthick=1, **next(styles_cycler))
        # ax.errorbar(delta_vals, data_list, label=data_name, capsize=5, capthick=1, **next(styles_cycler))

    ax.set_xlabel("Delta Value")
    ax.set_xticks([x/100.0 for x in range(9000,9999,66)])
    ax.set_ylim(-0.05,1.05)
    ax.set_ylabel("Adjusted Rand Index (ARI)")

    ax.legend(loc='lower left')

    if save:
        savename = graph_path+'DeltaAssumption.pdf'
        # print(savename)
        fig.savefig(savename, format='pdf', dpi=1200, bbox_inches='tight')

    else:
        plt.show()

def plotDetlaAssump_single(assumption_folder, graph_path, dataname="*_9*", metric="ari", save=False):
    delta_vals = [x/1000.0 for x in range(90000,99999,666)]

    fig = plt.figure(figsize=(18,12))
    ax = fig.add_subplot(111)

    # Extract all dataset names, then create a set, then randomly choose those
    # Then we can just plot and calculate what we need for each dataset at a time

    # Add dataname in here
    data_names = {i.split(os.sep)[-1].split("-")[0] for i in glob.glob(assumption_folder+os.sep+"*")}

    data_sets = random.sample(data_names, 7)

    # Maybe need enumerate just for style stuff?
    for dataset in data_sets:
        files = sorted(glob.glob(assumption_folder+os.sep+dataset+"*"+metric+"*"))

        data = [np.max(np.loadtxt(file, delimiter=","),axis=0) for file in files]

        assert len(data) == len(delta_vals)

        means = [np.mean(i) for i in data]
        stderrs = [np.std(i, ddof=0) for i in data]

        ax.errorbar(delta_vals, means, yerr=stderrs, label=dataset, capsize=5, capthick=1)

    ax.set_xticks(delta_vals)
    ax.set_xticklabels(['{0:.2f}'.format(delta) for delta in delta_vals])
    ax.legend(loc=3)
    plt.show()


def plotDeltaAssump_all(assumption_folder, graph_path, metric="ari", save=False):
    # Create a list of the delta values that we used
    # Then just loop over these, as we want to aggregate info for each delta value
    delta_vals = [x/1000.0 for x in range(90000,99999,666)]

    fig = plt.figure(figsize=(18,12))
    ax = fig.add_subplot(111)

    for dataname in ["*_9*","*UKC*"]:
        data = []

        for num_delta, delta in enumerate(delta_vals):
            print(delta)
            # Sort might be useful if we use this code to select the same random files to plot separately
            files = sorted(glob.glob(assumption_folder+os.sep+dataname+"*"+metric+"*"+str(delta)+"*"))

            for index, file in enumerate(files):
                if index == 0:
                    data_metric = np.max(np.loadtxt(file, delimiter=","),axis=0)

                else:
                    data_metric = np.append(data_metric, np.max(np.loadtxt(file, delimiter=","),axis=0))

            data.append(data_metric)

        assert len(data) == len(delta_vals)

        means = [np.mean(i) for i in data]
        stderrs = [np.std(i, ddof=0) for i in data]

        ax.errorbar(delta_vals, means, yerr=stderrs, capsize=5, capthick=1)

    ax.set_xticks(delta_vals)
    ax.set_xticklabels(['{0:.2f}'.format(delta) for delta in delta_vals], fontsize=20)
    ax.legend(loc=3, labels=["Synthetic","Real"])

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
    ax1.set_xlabel("Search Strategy")
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

        if metric_files == []:
            continue

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
        ax1.set_xlabel("Search Strategy")
        ax1.set_ylim(-0.05,1.05)

        ax2 = ax1.twinx()

        ax2.errorbar(list(range(len(strat_names))), means, color="black", linestyle="--", yerr=errs, capsize=7, capthick=1, label="Time")

        ax2.set_ylabel("Time")
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

        if metric_files == []:
            continue

        print(len(metric_files))

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
            data_time = (data_time - min_val)/denom

            means.append(np.mean(data_time))
            errs.append(np.std(data_time, ddof=0)/np.sqrt(data_time.shape[0]))

        # Plot the line here for each dataset
        # Similar method to above, basically mirroring the bars but with lines
        # ax2 = ax1.twinx()

        ax2.errorbar([i+2*dataset_num for i in ind[:len(means)]], means, color="red", linestyle="--", yerr=errs, capsize=7, capthick=1)
        # ax2.errorbar(list(range(len(strat_names))), means, color="black", linestyle="--", yerr=errs, capsize=7, capthick=1, label="Time")

    ax2.set_xticks(np.arange(num_subplots)*2+ind[num_subplots])

    plt.show()


def plotArtifExp_singlebox(dataset_folder,graph_path,metric="ari",save=False):
    metric_files = sorted(glob.glob(dataset_folder+os.sep+"*"+metric+"*"), reverse=True)
    time_files = sorted(glob.glob(dataset_folder+os.sep+"*"+"time"+"*"), reverse=True)
    print(metric_files)

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
    ax1.set_xlabel("Search Strategy")
    ax1.set_ylim(-0.05,1.05)

    ax2 = ax1.twinx()

    ax2.errorbar(list(range(1,len(data_metric_list)+1)), means, color="black", linestyle="--", yerr=errs, capsize=7, capthick=1, label="Time")

    ax2.set_ylabel("Time")

    ax2.set_xticklabels(strat_names)
    ax2.set_title("Comparison of time and performance for "+data_name, fontsize=22)
    ax2.set_ylim(-0.05,1.05)

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc=4)

    if save:
        savename = graph_path + "artif-interval-" + data_name + "-box.pdf"
        fig.savefig(savename, format='pdf', dpi=1200, bbox_inches='tight')
        plt.close(fig)
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
        # ax1.set_xlabel("Search Strategy")
        ax1.set_ylim(-0.05,1.05)

        ax2 = ax1.twinx()

        ax2.errorbar(list(range(1,len(data_metric_list)+1)), means, color="red", linestyle="--", yerr=errs, capsize=7, capthick=1)
        # ax2.errorbar(list(range(len(strat_names))), means, color="black", linestyle="--", yerr=errs, capsize=7, capthick=1, label="Time")

        # ax2.set_ylabel("Time")
        ax2.set_ylim(-0.05,1.05)
        # ax2.set_xticks(np.arange(len(strat_names)))
        ax2.set_xticklabels(strat_names)

    ax1.set_ylabel("ARI")
    ax1.set_xlabel("Search Strategy")
    ax2.set_ylabel("Time")

    axes = fig.get_axes()

    # -4 just means the bottom 2 with 7 dataset results
    for i in range(len(axes)-4):
        plt.setp(axes[i].get_xticklabels(), visible=False)

    plt.show()


    # Consider saving these axes into a dict or something, and then adding them to a fig afterwards with a gridspec subplot - may be able to control axes better/make it look better

def plotArtifExp_allDS(artif_folder, graph_path, strat_name_dict, dataname="*_9", metric="ari", method="random", save=False):
    # plt.rc('text', usetex=True)

    # Using *_9 just to select the new data, can modify to get the old
    folders = glob.glob(artif_folder+os.sep+dataname, recursive=True)

    fig = plt.figure(figsize=(18,12))
    ax1 = fig.add_subplot(111)
    
    # Have the strategies here in a defined order, then just check that the one extracted from the filename matches to ensure consistency
    stratname_ref = ["base-sr1", "base-sr5", "carryon", "fairmut", "hypermutall", "hypermutspec", "reinit"]

    # Lists to aggregate the data over all datasets
    data_metric_list = []
    data_time_list = []

    for num_dataset, dataset_folder in enumerate(folders):

        if dataname!="*UKC*":
            metric_files = glob.glob(dataset_folder+os.sep+"*base*"+metric+"*")
            metric_files.extend(glob.glob(dataset_folder+os.sep+"*"+metric+"*"+method+"*"))

            time_files = glob.glob(dataset_folder+os.sep+"*base*time*")
            time_files.extend(glob.glob(dataset_folder+os.sep+"*time*"+method+"*"))

        else:
            metric_files = glob.glob(dataset_folder+os.sep+"*"+metric+"*"+method+"*")
            time_files = glob.glob(dataset_folder+os.sep+"*time*"+method+"*")
            if method == "hv":
                metric_files.extend(glob.glob(dataset_folder+os.sep+"*base*"+metric+"*sr5*random*"))
                time_files.extend(glob.glob(dataset_folder+os.sep+"*base*time*sr5*random*"))

        print(metric_files, len(metric_files))

        # for new UKC results
        # time_files = glob.glob(dataset_folder+os.sep+"*time*"+method+"*")

        metric_files = sorted(metric_files, reverse=False)
        time_files = sorted(time_files, reverse=False)

        assert len(metric_files) == len(time_files)

        if metric_files == []:
            continue

        strat_names = []

        # Constants for normalising the data between 0 and 1
        min_val = np.inf
        max_val = 0

        # Extract data_name
        data_name = dataset_folder.split(os.sep)[-1]

        for index, file in enumerate(metric_files):	
            data_metric = np.max(np.loadtxt(file, delimiter=","),axis=0)

            if "base" in file:
                # strat_names.append("-".join([file.split(os.sep)[-1].split("-")[1].split("_")[-1],
                # 	file.split(os.sep)[-1].split("-")[-1].split(".")[0][:3]]))
                if dataname == "*UKC*":
                    strat_names.append("-".join([file.split(os.sep)[-1].split("-")[1],file.split(os.sep)[-1].split("-")[3]]))
                else:
                    strat_names.append("-".join([file.split(os.sep)[-1].split("-")[1],file.split(os.sep)[-1].split("-")[3][:-4]]))

            else:
                strat_names.append(file.split(os.sep)[-1].split("-")[1].split("_")[-1])

            assert strat_names[-1] == stratname_ref[index], "Strat name difference "+strat_names[-1]+" "+stratname_ref[index]

            # Create initial arrays for the first dataset, then append afterwards
            # The boxplot command can then handle everything
            # We should have just a single array for each of the strategies

            # strat_index = stratname_ref.index(strat_names[-1])
            # print(strat_names[-1], index, strat_index)

            # It could be useful to use stratname_ref.index(strat_names[-1]) to avoid enumerate for loop issue with empty datasets (though that shouldn't be a problem for the _9_ datasets)

            if num_dataset == 0:
                data_metric_list.append(data_metric)

            else:
                data_metric_list[index] = np.append(data_metric_list[index], data_metric)

            data_time = np.loadtxt(time_files[index], delimiter=',')
            # print(data_time)
            # print(strat_names[-1],"\n")

            if np.max(data_time) > max_val:
                max_val = np.max(data_time)

            if np.min(data_time) < min_val:
                min_val = np.min(data_time)
        
        # print(data_metric_list)
        print(len(data_metric_list))
        print(type(data_metric_list[0]))
        print(data_metric_list[0].shape)

        denom = max_val - min_val

        for index, file in enumerate(time_files):
            data_time = np.loadtxt(file, delimiter=',')
            data_time = (data_time - min_val)/denom

            if num_dataset == 0:
                data_time_list.append(data_time)

            else:
                # data_time_list[index].append(data_time)
                data_time_list[index] = np.append(data_time_list[index], data_time)

    # print(data_metric_list)
    assert len(data_metric_list) == len(stratname_ref)
    assert len(data_time_list) == len(stratname_ref)

    medians = []
    errs = []

    for times in data_time_list:
        medians.append(np.median(times))
        errs.append(np.std(times, ddof=0)/np.sqrt(times.shape[0]))

    # colors = []

    # # Might need to make an if synthetic for this, as it doens't quite work for UKC
    # for i, data in enumerate(data_metric_list):
    # 	if "sr1" in stratname_ref[i]:
    # 		colors.append("white")
    # 		continue

    # 	elif "sr5" in stratname_ref[i]:
    # 		colors.append("dimgray")
    # 		continue

    # 	else:
    # 		sum_ranks, p_val = wilcoxon(data_metric_list[0], data_metric_list[i], zero_method='wilcox')
    # 		print(stratname_ref[1], stratname_ref[i])
    # 		print(sum_ranks, p_val,"\n")

    # 		# if dataname=="*UKC*":
    # 		# 	sum_ranks, p_val = wilcoxon(data_metric_list[0], data_metric_list[i], zero_method='wilcox')
    # 		# 	print(stratname_ref[0], stratname_ref[i])
    # 		# 	print(sum_ranks, p_val,"\n")

    # 		if p_val >= 0.05:
    # 			colors.append("dimgray")
    # 		else:
    # 			colors.append("white")

    medianprops = dict(linewidth=2, color='midnightblue')

    bxplot = ax1.boxplot(data_metric_list, patch_artist=True, medianprops=medianprops)

    # for patch, color in zip(bxplot['boxes'],colors):
    # 	patch.set_facecolor(color)

    for patch in bxplot['boxes']:
        patch.set_facecolor("None")

    ax1.set_ylabel("Adjusted Rand Index (ARI)")
    ax1.set_xlabel("Search Strategy")

    if dataname == "*UKC*":
        ax1.set_ylim(0.75,1.01)
    else:
        ax1.set_ylim(0.35,1.05)

    ax2 = ax1.twinx()

    ax2.errorbar(list(range(1,len(data_time_list)+1)), medians, color="darkred", linestyle="--", yerr=errs, capsize=7, capthick=1, label="Time")

    ax2.set_ylabel("Standarised Time per Run")
    ax2.set_ylim(-0.05,1.05)

    # if "*_9" in dataname:
    # 	ax2.set_title("Synthetic Datasets with "+method+" trigger", fontsize=30)
    # elif "UKC" in dataname:
    # 	ax2.set_title("Real Datasets with "+method+" trigger", fontsize=30)
    # else:
    # 	ax2.set_title("All Datasets with "+method+" trigger", fontsize=30)

    for i, strat in enumerate(stratname_ref):
        stratname_ref[i] = strat_name_dict[strat]

    # handles, labels = ax2.get_legend_handles_labels()

    ax2.set_xticklabels(stratname_ref)#, fontsize=50)
    ax2.legend(loc=4)

    if save:
        if "*_9" in dataname:
            savename = graph_path + "artif-synthds-"+method+"-box.pdf"
        elif "UKC" in dataname:
            savename = graph_path + "artif-realds-"+method+"-box.pdf"
        else:
            savename = graph_path + "artif-allds-"+method+"-box.pdf"
        fig.savefig(savename, format='pdf', dpi=1200, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
    return ax2

def plotArtif_specStrat(results_path, strategy="reinit"):
    fnames = []

    fnames.extend(glob.glob(results_path+os.sep+"*base*sr5*"))
    fnames.extend(sorted(glob.glob(results_path+os.sep+"*"+strategy+"*"), reverse=True))

    data = []

    strat_names = []

    for fname in fnames:
        data.append(np.loadtxt(fname, delimiter=","))

        strat_names.append("-".join([fname.split(os.sep)[-1].split(".")[0].split("-")[-2],fname.split(os.sep)[-1].split(".")[0].split("-")[-1]]))

    fig = plt.figure(figsize=(18,12))
    ax1 = fig.add_subplot(111)
    ax1.boxplot(data)
    ax1.set_xticklabels(strat_names)

    plt.show()

def plotArtif_pairs(results_path, strategy="reinit"):
    fnames = []
    fnames.extend(glob.glob(results_path+os.sep+"*base*sr5*"))
    fnames.extend(sorted(glob.glob(results_path+os.sep+"*"+strategy+"*"), reverse=True))

    data = []

    strat_names = []

    for fname in fnames:
        data.append(np.loadtxt(fname, delimiter=","))

        strat_names.append("-".join([fname.split(os.sep)[-1].split(".")[0].split("-")[-2],fname.split(os.sep)[-1].split(".")[0].split("-")[-1]]))


    d = len(data)
    fig, axes = plt.subplots(nrows=d, ncols=d)#, sharex='col', sharey='row')
    a=0
    b=1
    for i in range(d):
        for j in range(d):
            ax = axes[i,j]
            if i == j:
                # ax.text(0.5, 0.5, strat_names[i], transform=ax.transAxes,
                # 		horizontalalignment='center', verticalalignment='center',
                # 		fontsize=16)
                ax.hist(data[i])
                # ax.set_title(strat_names[i])
            else:
                for start_ind in range(0,1050,30):
                    ax.scatter(data[j][start_ind:start_ind+29], data[i][start_ind:start_ind+29], s=6)

            if i == 0:
                ax.set_title(strat_names[a], fontsize=18)
                a+=1
            if j == 0 and i != 0:
                # title = ax.set_title(strat_names[b], fontsize=18)
                # y_ax_label = ax.set_ylabel("")
                # title.set_position(y_ax_label.get_position() + (-2.75,0))
                # title.set_rotation(90)
                ax.set_title(strat_names[b], rotation='vertical',x=-0.25,y=0.8, fontsize=18)
                b+=1

    plt.show()

def plotArtif_pairs2(results_path, strategy="reinit"):
    fnames = []
    fnames.extend(glob.glob(results_path+os.sep+"*base*sr5*"))
    fnames.extend(sorted(glob.glob(results_path+os.sep+"*"+strategy+"*"), reverse=True))

    data = []

    strat_names = []

    for fname in fnames:
        data.append(np.loadtxt(fname, delimiter=","))

        strat_names.append("-".join([fname.split(os.sep)[-1].split(".")[0].split("-")[-2],fname.split(os.sep)[-1].split(".")[0].split("-")[-1]]))

    interval_triggers = [[12, 40, 63, 87], [12, 40, 64, 80], [27, 41, 60, 80], [13, 38, 60, 75], [19, 31, 64, 75], [11, 41, 51, 71], [21, 39, 64, 74], [10, 32, 50, 75], [23, 31, 50, 73], [29, 39, 66, 70], [23, 31, 52, 78], [20, 44, 50, 82], [11, 34, 55, 78], [22, 42, 64, 72], [11, 39, 60, 72], [28, 49, 66, 74], [20, 31, 67, 89], [27, 36, 66, 73], [14, 30, 57, 78], [12, 42, 65, 79], [18, 33, 50, 88], [14, 36, 51, 83], [20, 42, 61, 74], [25, 48, 56, 70], [10, 43, 64, 83], [20, 36, 60, 74], [26, 44, 61, 81], [10, 44, 66, 88], [29, 45, 57, 76], [14, 31, 54, 87]]

    interval_triggers = [i[-1] for i in interval_triggers]

    random_triggers = [[33, 43, 53, 84], [13, 45, 62, 72], [32, 54, 76, 88], [9, 71, 81, 91], [18, 28, 39, 49], [42, 52, 62, 72], [14, 24, 59, 77], [21, 40, 50, 60], [14, 27, 74, 89], [49, 60, 70, 80], [17, 27, 47, 73], [52, 62, 72, 82], [11, 21, 31, 87], [11, 21, 82, 92], [19, 29, 55, 66], [32, 42, 67, 81], [15, 78, 89, 99], [11, 36, 46, 56], [35, 60, 70, 80], [54, 74, 84, 94], [24, 34, 72, 82], [42, 52, 64, 74], [15, 30, 57, 67], [14, 34, 44, 84], [13, 27, 44, 73], [25, 35, 45, 55], [43, 53, 63, 73], [29, 39, 49, 59], [49, 59, 69, 79], [24, 51, 61, 81]]

    random_triggers = [i[-1] for i in random_triggers]

    d = len(data)
    fig, axes = plt.subplots(nrows=d, ncols=d)#, sharex='col', sharey='row')
    a=0
    b=1
    for i in range(d):
        for j in range(d):
            ax = axes[i,j]
            if i == j:
                # ax.text(0.5, 0.5, strat_names[i], transform=ax.transAxes,
                # 		horizontalalignment='center', verticalalignment='center',
                # 		fontsize=16)
                ax.hist(data[i])
                # ax.set_title(strat_names[i])
            else:
                for ind, start_ind in enumerate(range(0,1050,30)):
                    # print(ind)
                    if "random" in strat_names[i]:
                        ax.scatter(data[j][start_ind:start_ind+29], data[i][start_ind:start_ind+29], c=[str(i/100) for i in random_triggers], s=6)

                    elif "interval" in strat_names[i]:
                        ax.scatter(data[j][start_ind:start_ind+29], data[i][start_ind:start_ind+29], c=[str(i/100) for i in interval_triggers], s=6)
                    else:
                        ax.scatter(data[j][start_ind:start_ind+29], data[i][start_ind:start_ind+29], s=6)


            if i == 0:
                ax.set_title(strat_names[a], fontsize=18)
                a+=1
            if j == 0 and i != 0:
                # title = ax.set_title(strat_names[b], fontsize=18)
                # y_ax_label = ax.set_ylabel("")
                # title.set_position(y_ax_label.get_position() + (-2.75,0))
                # title.set_rotation(90)
                ax.set_title(strat_names[b], rotation='vertical',x=-0.25,y=0.8, fontsize=18)
                b+=1

    plt.show()

def plotArtif_allDS_multifig(artif_folder, strat_name_dict, methods, dataname="*_9", metric="ari", save=False):

    folders = glob.glob(artif_folder+os.sep+dataname, recursive=True)
    # folders = folders[:13]

    # fig = plt.figure(figsize=(18,12))

    # Have the strategies here in a defined order, then just check that the one extracted from the filename matches to ensure consistency

    fig, (ax1, ax2, ax3) = plt.subplots(1,3, sharey=True)#, figsize=(18,12))

    axes_list = [ax1, ax2, ax3]

    for num_subplot, method in enumerate(methods):
        # ax1 = fig.add_subplot(1,3,num_subplot+1, sharey=)

        # Lists to aggregate the data over all datasets
        data_metric_list = []
        data_time_list = []

        stratname_ref = ["base-sr1", "base-sr5", "carryon", "fairmut", "hypermutall", "hypermutspec", "reinit"]
        
        for num_dataset, dataset_folder in enumerate(folders):
            metric_files = glob.glob(dataset_folder+os.sep+"*base*"+metric+"*")
            metric_files.extend(glob.glob(dataset_folder+os.sep+"*"+metric+"*"+method+"*"))

            time_files = glob.glob(dataset_folder+os.sep+"*base*time*")
            time_files.extend(glob.glob(dataset_folder+os.sep+"*time*"+method+"*"))

            metric_files = sorted(metric_files, reverse=False)
            time_files = sorted(time_files, reverse=False)
            # print(metric_files)

            assert len(metric_files) == len(time_files)

            if metric_files == []:
                continue

            strat_names = []

            # Constants for normalising the data between 0 and 1
            min_val = np.inf
            max_val = 0

            # Extract data_name
            data_name = dataset_folder.split(os.sep)[-1]

            for index, file in enumerate(metric_files):
                # print(file)
                
                data_metric = np.max(np.loadtxt(file, delimiter=","),axis=0)

                if "base" in file:
                    # strat_names.append("-".join([file.split(os.sep)[-1].split("-")[1].split("_")[-1],
                    # 	file.split(os.sep)[-1].split("-")[-1].split(".")[0][:3]]))

                    strat_names.append("-".join([file.split(os.sep)[-1].split("-")[1],file.split(os.sep)[-1].split("-")[3][:-4]]))

                else:
                    strat_names.append(file.split(os.sep)[-1].split("-")[1].split("_")[-1])

                # Show order of strategies
                # print(strat_names[-1], index, stratname_ref[index])

                assert strat_names[-1] == stratname_ref[index], "Strat name difference "+strat_names[-1]+" "+stratname_ref[index]

                # Create initial arrays for the first dataset, then append afterwards
                # The boxplot command can then handle everything
                # We should have just a single array for each of the strategies

                # strat_index = stratname_ref.index(strat_names[-1])
                # print(strat_names[-1], index, strat_index)

                # It could be useful to use stratname_ref.index(strat_names[-1]) to avoid enumerate for loop issue with empty datasets (though that shouldn't be a problem for the _9_ datasets)

                if num_dataset == 0:
                    data_metric_list.append(data_metric)

                else:
                    data_metric_list[index] = np.append(data_metric_list[index], data_metric)

                data_time = np.loadtxt(time_files[index], delimiter=',')

                if np.max(data_time) > max_val:
                    max_val = np.max(data_time)

                if np.min(data_time) < min_val:
                    min_val = np.min(data_time)

            denom = max_val - min_val

            for index, file in enumerate(time_files):
                data_time = np.loadtxt(file, delimiter=',')
                data_time = (data_time - min_val)/denom

                if num_dataset == 0:
                    data_time_list.append(data_time)

                else:
                    # data_time_list[index].append(data_time)
                    data_time_list[index] = np.append(data_time_list[index], data_time)

        assert len(data_metric_list) == len(stratname_ref)
        assert len(data_time_list) == len(stratname_ref)

        medians = []
        errs = []

        for times in data_time_list:
            medians.append(np.median(times))
            errs.append(np.std(times, ddof=0)/np.sqrt(times.shape[0]))

        axes_list[num_subplot].boxplot(data_metric_list)
        # axes_list[num_subplot].set_ylabel("Adjusted Rand Index (ARI)")
        axes_list[num_subplot].set_xlabel("Search Strategy")
        axes_list[num_subplot].set_ylim(0.35,1.05)

        ax_y = axes_list[num_subplot].twinx()

        ax_y.errorbar(list(range(1,len(data_time_list)+1)), medians, color="red", linestyle="--", yerr=errs, capsize=7, capthick=1, label="Time")

        # ax_y.set_ylabel("Standarised Time per Run")
        ax_y.set_ylim(-0.05,1.05)

        # ax_y.set_title("Comparison of strategies over all datasets using "+method+" trigger", fontsize=22)

        for i, strat in enumerate(stratname_ref):
            stratname_ref[i] = strat_name_dict[strat]

        ax_y.set_xticklabels(stratname_ref)
        plt.setp(axes_list[num_subplot].get_xticklabels(), rotation=60, horizontalalignment='right')
        ax_y.legend(loc=4)

    ax1.set_ylabel("Adjusted Rand Index (ARI)")
    ax_y.set_ylabel("Standarised Time per Run")
    # plt.tight_layout()

    if save:
        savename = graph_path + "artif-allds-box.pdf"
        fig.savefig(savename, format='pdf', dpi=1200, bbox_inches='tight')
        plt.close(fig)

    else:
        plt.show()

def plotArtif_HV(artif_folder, strat="reinit", dataname="*_9", save=False):
    # Filter to ensure only directories are returned
    folders = filter(os.path.isdir,glob.glob(artif_folder+os.sep+dataname))
    print(folders)

    line_names = [r'$\Delta\operatorname{-MOCK} (sr5)$', r'$\mathit{RO}-HV$', r'$\mathit{RO}-Interval$', r'$\mathit{RO}-Random$']

    run_num = 11

    styles = [
    {'color':'k', 'dashes':(None,None), 'marker':"None",'ms':7},
    {'color':'c', 'dashes':(5,2), 'marker':"None",'ms':7},
    {'color':'m', 'dashes':(2,5), 'marker':"None",'ms':7},
    {'color':'g', 'dashes':(2,1,2,1), 'marker':"None",'ms':7}]

    # Use None to help with legend creation loop
    markers = ["None", "o", "^", "D"]
    # dashes = [(None,None),(5,2),(2,5),(3,1,3)]
    linewidth = 3.5

    styles_cycler = cycle(styles)

    for folder in folders:
        print(folder)
        # Maybe add sr1 after to see how it looks?
        files = glob.glob(folder+os.sep+"*base*hv*sr5*")
        files.extend(sorted(glob.glob(folder+os.sep+"*"+strat+"-hv-*")))

        # print(len(files))
        assert len(files) == 4

        data_list = [np.loadtxt(file, delimiter=",", usecols=run_num) for file in files]

        fig = plt.figure(figsize=(18,12))
        ax = fig.add_subplot(111)

        trigger_files = sorted(glob.glob(folder+os.sep+"*"+strat+"-triggers-*"))
        # trigger_list = [np.loadtxt(file, delimiter=",") for file in trigger_files]

        trigger_list = []

        for file in trigger_files:
            trigger_list.append([list(map(int,line.split(","))) for line in open(file)][run_num][1:])

        print(len(trigger_list))

        for i, data in enumerate(data_list):
            # print(data, data.shape)
            ax.errorbar(list(range(0,data.shape[0])), data,
                label=line_names[i], linewidth=linewidth,
                **next(styles_cycler))

            # No triggers for base
            if i>0:
                ax.plot(trigger_list[i-1], data[trigger_list[i-1]], linestyle='None',
                    marker=markers[i], ms=15,
                    color=styles[i]['color'],label=line_names[i])

        ax.set_xlabel("Generations")
        ax.set_ylabel("Hypervolume")

        handles = []
        for i, line in enumerate(line_names):
            handles.append(mlines.Line2D([],[], color=styles[i]['color'], dashes=styles[i]['dashes'], marker=markers[i], ms=15, linewidth=linewidth, label=line))

        ax.legend(handles=handles, labels=line_names)

        if save:
            savename = graph_path + "artif-" + folder.split(os.sep)[-1] + "-hvplot-run" +str(run_num)+".pdf"
            fig.savefig(savename, format='pdf', dpi=1200, bbox_inches='tight')
        else:
            plt.show()

        plt.close(fig)
        print("\n")

def plotArtifClusts_all(artif_folder, graph_path, strat_name_dict, dataname="*UKC*", metric="numclusts", method="random", save=False):
    # plt.rc('text', usetex=True)

    # Using *_9 just to select the new data, can modify to get the old
    folders = sorted(glob.glob(artif_folder+os.sep+dataname, recursive=True))
    # folders = folders[:13]
    print(folders)

    # Have the strategies here in a defined order, then just check that the one extracted from the filename matches to ensure consistency
    stratname_ref = ["base-sr1", "base-sr5", "carryon", "fairmut", "hypermutall", "hypermutspec", "reinit"]

    for num_dataset, dataset_folder in enumerate(folders):
        dataset_name = dataset_folder.split(os.sep)[-1]

        metric_files = glob.glob(dataset_folder+os.sep+"*base*"+metric+"*")
        metric_files.extend(glob.glob(dataset_folder+os.sep+"*"+metric+"*"+method+"*"))
        metric_files = sorted(metric_files, reverse=False)

        ari_files = glob.glob(dataset_folder+os.sep+"*base*ari*")
        ari_files.extend(glob.glob(dataset_folder+os.sep+"*ari*"+method+"*"))
        ari_files = sorted(ari_files, reverse=False)

        if metric_files == []:
            continue

        strat_names = []
        data_metric_list = []

        fig = plt.figure(figsize=(18,12))
        ax1 = fig.add_subplot(111)

        # Extract data_name
        data_name = dataset_folder.split(os.sep)[-1]

        for index, file in enumerate(metric_files):	
            data_metric = np.loadtxt(file, delimiter=",")
            # Select the number of clusters represented by the best 30 ARIs
            # best_ari_indices = np.argmax(np.loadtxt(ari_files[index], delimiter=","),axis=0)
            # data_metric = data_metric.T[np.arange(len(data_metric.T)),best_ari_indices]

            if "base" in file:
                # strat_names.append("-".join([file.split(os.sep)[-1].split("-")[1].split("_")[-1],
                # 	file.split(os.sep)[-1].split("-")[-1].split(".")[0][:3]]))

                strat_names.append("-".join([file.split(os.sep)[-1].split("-")[1],file.split(os.sep)[-1].split("-")[3][:-4]]))

            else:
                strat_names.append(file.split(os.sep)[-1].split("-")[1].split("_")[-1])

            # Show order of strategies

            assert strat_names[-1] == stratname_ref[index], "Strat name difference "+strat_names[-1]+" "+stratname_ref[index]

            # It could be useful to use stratname_ref.index(strat_names[-1]) to avoid enumerate for loop issue with empty datasets (though that shouldn't be a problem for the _9_ datasets)

            data_metric_list.append(data_metric)

        assert len(data_metric_list) == len(stratname_ref)

        medianprops = dict(linewidth=2, color='midnightblue')

        bxplot = ax1.boxplot(data_metric_list, patch_artist=True, medianprops=medianprops)

        # raw_fname = "/home/cshand/Documents/Delta-MOCK/data/synthetic_datasets/" + dataset_name + "_labels_headers.data"
        raw_fname = "/home/cshand/Documents/Delta-MOCK/data/UKC_datasets/" + dataset_name + ".txt"

        with open(raw_fname) as file:
            head = [int(next(file)[:-1]) for _ in range(4)]

        true_clusts = head[3]
        print(dataset_name, true_clusts)

        for patch in bxplot['boxes']:
            patch.set_facecolor("None")

        ax1.plot(list(range(0,len(strat_names)+2)), [int(true_clusts)]*(len(strat_names)+2), linestyle = "--", label="True no. clusters", color="darkred")

        ax1.set_ylabel("Number of Clusters")
        ax1.set_xlabel("Search Strategy")
        ax1.set_ylim(0,600)
        # ax1.set_title(dataset_name)

        for i, strat in enumerate(stratname_ref):
            strat_names[i] = strat_name_dict[strat]

        ax1.set_xticklabels(strat_names, fontsize=24)
        ax1.legend(loc=2)

        if save:
            savename = graph_path + "artif-"+dataset_name+"-numclusts-"+method+"-box.pdf"
            fig.savefig(savename, format='pdf', dpi=1200, bbox_inches='tight')
            plt.close(fig)

        else:
            plt.show()
            plt.close(fig)

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

    # dataset_folders = glob.glob(results_path+os.sep+"*")
    # dataset_folders.remove(graph_path)
    # dataset_folders.remove(artif_folder)

    # font = {'family' : 'normal',
    # 	'weight' : 'medium',
    # 	'size'   : 12}

    # plt.rc('font', **font)

    SMALL_SIZE = 26
    MEDIUM_SIZE = 30
    BIGGER_SIZE = 24
    # plt.rc('text', usetex=True)
    plt.rc('font', size=MEDIUM_SIZE, family='serif')          # controls default text sizes
    # plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    # plt.rc('font', family='serif')

    plt.rc('mathtext', fontset='cm')

    artif_folder = os.path.join(results_path, "artif")+os.sep
    graph_path = os.path.join(results_path, "graphs")+os.sep

    # for dataset_folder in dataset_folders:
    # 	plotArtifExp_singlebox(dataset_folder,graph_path,metric="ari",save=False)

    # plotArtifExp_multiple(artif_folder)
    # plotArtifExp_multiple2(artif_folder)
    # plotArtifExp_multiplebox(artif_folder)

    # plotDeltaAssump(assumption_folder,graph_path,styles_cycler)

    # Remaps strategy names with mathtext formatting
    # Will need to add ones for base-MOCK at SR5 and SR1
    strat_name_dict = {
    "base-sr1" : r'$\Delta{-}MOCK$' '\n' r'($\delta_{High}$)', 
    "base-sr5" : r'$\Delta{-}MOCK$' '\n' r'($\delta_{Low}$)',
    "carryon" : r'$\mathit{CO}$',
    "fairmut" : r'$\mathit{FM}$',
    "hypermutall" : r'$\mathit{TH}_{all}$',
    "hypermutspec" : r'$\mathit{TH}_{new}$',
    "reinit" : r'$\mathit{RO}$',
    }

    # plotArtif_specStrat(results_path)
    # plotArtif_pairs(results_path)
    # plotArtif_pairs2(results_path)

    methods = ["random", "interval", "hv"]

    # plotArtif_allDS_multifig(artif_folder, strat_name_dict, methods)

    plotArtifExp_allDS(artif_folder, graph_path, strat_name_dict, dataname="*_9", method="interval", save=False)
    # plotArtifExp_allDS(artif_folder, graph_path, strat_name_dict, dataname="*UKC*", method="interval", save=False)

    # for method in methods:
        # plotArtifExp_allDS(artif_folder, graph_path, strat_name_dict, dataname="*UKC*", method=method, save=False)
        # plotArtifExp_allDS(artif_folder, graph_path, strat_name_dict, dataname="*_9", method=method, save=False)
        # plotArtifClusts_all(artif_folder, graph_path, strat_name_dict,method=method, dataname="*UKC*", save=True)

    # plotDeltaAssump_all(assumption_folder, graph_path)
    # plotDetlaAssump_single(assumption_folder, graph_path)

    # plotArtif_HV(artif_folder, dataname="*UKC*",save=False)

    ### TO DO ###

    # Clean this up by using separate functions to load and format the data (i.e. extract best ARI from each run)
    # Then we can just pass it to the relevant plotting function and reduce this code hugely
