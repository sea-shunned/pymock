import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# try to construct this using the pathlib module

def get_folders(folder_str):
    results_folder = Path.cwd() / "results"
    
    # for folder in results_folder.glob(folder_str+"*"):
    #     print(folder)

    print(results_folder.glob(folder_str / "*"))

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
        sum_ranks, p_val = stats_func(data_1, data_2)
        return sum_ranks, p_val
    else:
        raise ValueError(f"Whoopsie! Behaviour not yet implemented...")

    # Asking for a general func may be too much - the below might even fail with some funcs
    # scipy.stats don't have unified function behaviour (obvs)
    # return stats_func(data_1, data_2)

def stats_colours(data, params):
    colours = []

    for data_set in data:
        pass


# would having one big container be useful?
# like a dictionary where the keys are the different sets of results
# then have sub_dicts with name, if it's the reference, actual data etc.

results = {

}

def aggreg_data(fpaths):
    # Set up empty list to hold the data
    data = []

    # Loop over the files
    for index, file in enumerate(fpaths):
        # Append the data to the list
        # Load in the data 
        # maybe add if statement here so we only use the max if measure is ari
        data.append(np.max(np.loadtxt(file, delimiter=","), axis=0))

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
            tick_labels = [r"$Orig$"]
        else:
            bplot_data = []
            tick_labels = []

        exp_folder = res_folder / params['mut_method']

        # get the individual folders of results
        exp_folders = sorted([x for x in exp_folder.iterdir() if x.is_dir()])
        
        for l_folder in exp_folders:
            data_files = get_fpaths(l_folder, glob_str=params['file_glob_str'])
            bplot_data.append(aggreg_data(data_files))
            tick_labels.append(f"L{str(l_folder)[-1]}")
            
    return bplot_data, tick_labels

def gen_graph_obj(params, nrows=1, ncols=1):
    fig, ax = plt.subplots(nrows, ncols, figsize=params['figsize'])
    return fig, ax


def main(params):
    results_folder = base_res_folder(params['exp_name'])
    print(results_folder)

    bplot_data, tick_labels = get_bplot_data(results_folder, params)

    # Generate a graph
    fig, ax = gen_graph_obj(params)

    ax = plot_boxplot(bplot_data, ax, tick_labels, params)

    if params['save_fig']:
        pass
    else:
        plt.show()

if __name__ == '__main__':
    params = {
        'exp_name': "mut_ops",
        'mut_method': "centroid",
        'group_by': "L",
        'file_glob_str': "*80*ari*",
        'xlabel': "L values",
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
        'figsize': (10,6),
        'save_fig': False
    }

    if params['stats_test']:
        # Define colours for stats test results
        params['colours'] = {
            'better': '#158915',
            'worse': '#AC5C1A',
            'equal': "dimgray",
            'reference': 'white'
        }
        # Define hatching for these too
        params['hatches'] = {
            'better': "/",
            'worse': '\\',
            'equal': "",
            'reference': ''            
        }
    # else:
    #     params['colours'] = {

    #     }

    main(params)