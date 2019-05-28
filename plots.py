import os
import glob
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats

import rpy2
import readline # fixes problem with rpy2
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr

# def get_folders(folder_str):
#     results_folder = Path.cwd() / "results"
    
#     # for folder in results_folder.glob(folder_str+"*"):
#     #     print(folder)

#     print(results_folder.glob(folder_str / "*"))

def base_res_folder(exp_name):
    return Path.cwd() / "results" / exp_name

def get_fpaths(res_folder, glob_str="*"):
    # Get some file paths
    # Could have string to select subset
    # also arg that specifies folders
    # do we want a list (or lis tof lists, where each sublist is a folder)?
    # or just handle single folders
    # pass

    if not res_folder.is_dir():
        raise ValueError(f"{res_folder} is not a directory!")

    f_paths = [str(fname) for fname in res_folder.glob(glob_str)]

    return f_paths

def plot_boxplot(data, ax, tick_labels, params):
    # can use **kwargs for mpl props
    # May take some fiddling
    # pass

    # medianprops = dict(linewidth=2, color='midnightblue')

    bxplot = ax.boxplot(data, **params['boxplot_kwargs'])

    if params['stats_test']:
        for patch, colour, hatch in zip(
            bxplot['boxes'], params['colours'], params['hatches']):

            patch.set_facecolor(colour)
            patch.set_hatch(hatch)

    ax.set_xlabel(params['xlabel'])
    ax.set_ylabel(params['ylabel'])

    ax.set_title(f"{params['mut_method'][0].upper()+params['mut_method'][1:]} mutation method")

    ax.set_xticklabels(tick_labels)

    return ax

def save_graph(fig_obj, save_path):
    # plt.savefig(save_path, etc.)
    pass

def norm_data(data):
    # function to normalise data
    # maybe need this
    # prev used when normalising time taken data
    pass

def stats_test(data_1, data_2, test_name="wilcoxon"):
    try:
        stats_func = getattr(stats, test_name)
    # sum_ranks, p_val = wilxocon(data_1, data_2)
    except AttributeError:
        raise(f"Cannot find {test_name} in scipy.stats!")

    if test_name == "wilcoxon":
        # print(data_1.shape, data_2.shape)
        sum_ranks, p_val = stats_func(x=data_1, y=data_2, zero_method='wilcox')
        return sum_ranks, p_val
    else:
        raise ValueError(f"Whoopsie! Behaviour not yet implemented...")

    # Asking for a general func may be too much - the below might even fail with some funcs
    # scipy.stats don't have unified function behaviour (obvs)
    # return stats_func(data_1, data_2)

def stats_colours(data_list, params):
    colours = []
    hatches = []
    
    for i, data in enumerate(data_list):
        sum_ranks, p_val = stats_test(data_list[0], data)

        if p_val >= 0.05:
            colours.append(params['stats_colours']['equal'])
            hatches.append(params['stats_hatches']['equal'])
        else:
            d = data_list[0] - data
            d = np.compress(np.not_equal(d,0),d,axis=-1)
            r = stats.rankdata(abs(d))
            r_plus = np.sum((d>0)*r, axis=0)
            r_minus = np.sum((d<0)*r, axis=0)

            assert min(r_plus, r_minus) == sum_ranks, "Sum rank calculation error!"

            if r_plus > r_minus:
                colours.append(params['stats_colours']['worse'])
                hatches.append(params['stats_hatches']['worse'])
            else:
                colours.append(params['stats_colours']['better'])
                hatches.append(params['stats_hatches']['better'])

    params['colours'] = colours
    params['hatches'] = hatches
    return params

def aggreg_data(fpaths):
    # Set up empty list to hold the data
    data = []

    # Loop over the files
    for index, file in enumerate(fpaths):
        # Append the data to the list
        # Load in the data 
        # maybe add if statement here so we only use the max if measure is ari

        if params['aggreg'] == "mean":
            data.append(np.mean(np.loadtxt(file, delimiter=","), axis=0))
        elif params['aggreg'] == "max":
            data.append(np.max(np.loadtxt(file, delimiter=","), axis=0))
        elif params['aggreg'] == "all":
            res = np.loadtxt(file, delimiter=",")
            shp = res.shape
            data.append(res.reshape(shp[0]*shp[1],))            
        else:
            raise ValueError(f"{params['aggreg']} aggregation method not implemented!")


    # Concatenate the data together for the boxplot
    final_data = np.concatenate(data, axis=0)
    return final_data


def get_bplot_data(res_folder, params):
    if params['mut_method'] == "all":
        pass
    else:
        if params['show_orig']:
            orig_fpaths = get_fpaths(res_folder / "orig", glob_str=params['file_glob_str'])
            bplot_data = [aggreg_data(orig_fpaths)]
            tick_labels = ["Orig"]
        else:
            bplot_data = []
            tick_labels = []

        exp_folder = res_folder / params['mut_method']

        # get the individual folders of results
        exp_folders = [x for x in exp_folder.iterdir() if x.is_dir()]
        # Sort them specifically by the numerical value of L
        exp_folders = sorted(exp_folders, key = lambda x: int(str(x).split(os.sep)[-1][1:]))

        for l_folder in exp_folders:
            data_files = get_fpaths(l_folder, glob_str=params['file_glob_str'])
            bplot_data.append(aggreg_data(data_files))
            tick_labels.append(f"{str(l_folder).split(os.sep)[-1]}")
            
    return bplot_data, tick_labels


def gen_graph_obj(params, nrows=1, ncols=1):
    fig, ax = plt.subplots(nrows, ncols, figsize=params['figsize'])
    return fig, ax


def get_eaf_data(params, results_folder):
    pass

def r_setup():
    eaf = importr('eaf', lib_loc="/home/cshand/R/x86_64-pc-linux-gnu-library/3.4")

    ploteaf = robjects.r['plotEAF']

    return ploteaf


def main(params):
    results_folder = base_res_folder(params['exp_name'])
    print(results_folder)

    # Generate the graph fig
    fig, ax = gen_graph_obj(params)

    if params['type'] == "bplot":
        bplot_data, tick_labels = get_bplot_data(results_folder, params)

        if params['stats_test']:
            params = stats_colours(bplot_data, params)

        ax = plot_boxplot(bplot_data, ax, tick_labels, params)
    
    elif params['type'] == "eaf":
        get_eaf_data(params)
    
    else:
        raise ValueError(f"{params['type']} has not been implemented!")

    if params['save_fig']:
        graph_path = params['graph_path'] / params['exp_name']
        try:
            os.makedirs(graph_path)
        except FileExistsError:
            pass
        
        savename = str(graph_path) + os.sep + "-".join([params['type'], params['file_glob_str'], params['mut_method'], params['aggreg']]) + ".pdf"
        
        savename = savename.replace("*","")

        if os.path.isfile(savename):
            print(f"Overwriting {savename}")

        fig.savefig(savename, format='pdf', dpi=1200, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

if __name__ == '__main__':
    params = {
        'exp_name': "mut_ops",
        'mut_method': "centroid",
        'group_by': "L",
        'file_glob_str': "*numclusts*",
        'type': "bplot",
        'exp_name': "mut_ops",
        'mut_method': "centroid",
        'aggreg': "all",
        'file_glob_str': "*ari*",
        'xlabel': "L* values",
        'ylabel': "Adjusted Rand Index (ARI)",
        'show_orig': True,
        'colours': None,
        'stats_test': False,
        'boxplot_kwargs': {
            'medianprops': {
                'linewidth': 2,
                'color': 'black',
                'solid_capstyle': "butt"
                },
            'patch_artist': True
        },
        'stats_test': True,
        'figsize': (18,12),
        'save_fig': False,
        'graph_path': Path.cwd() / "results" / "graphs"
    }

    if params['stats_test']:
        # Define colours for stats test results
        params['stats_colours'] = {
            'better': '#158915',
            'worse': '#AC5C1A',
            'equal': "dimgray",
            'reference': 'white'
        }
        # Define hatching for these too
        params['stats_hatches'] = {
            'better': "/",
            'worse': '\\',
            'equal': "",
            'reference': ""            
        }

    if params['type'] == "bplot":
        params['boxplot_kwargs'] = {
            'medianprops': {
                'linewidth': 2,
                'color': 'black',
                'solid_capstyle': "butt"
                },
            'patch_artist': True
        }

    if params['type'] == "eaf":
        params["left_label"] = "Original"
        params["left_method"] = "orig"
        params["left_L"] = "" # "" if "orig", otherwise whatever

        params["right_label"] = "Centroid"
        params["right_method"] = "centroid"
        params["right_L"] = 5

    plt.style.use('seaborn-paper')
    SMALL_SIZE = 28
    MEDIUM_SIZE = 30
    BIGGER_SIZE = 30
    # plt.rc('text', usetex=True)
    plt.rc('font', size=MEDIUM_SIZE, family='serif') # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    # plt.rc('font', family='serif')

    main(params)

    # Need to add option to aggregate by max, mean, or use all
    # Then add this to the filename
