This document serves as a brief user guide, outlining the contents of each of the files and giving some pointers for how certain parts of these files join together.

# File Guide
* `classes.py`: A terribly named script with the main classes for this implementation. These classes are as follows:
    * `PartialClust`- Class for the components, mainly used for the delta-evaluation.
    * `Datapoint`- Class for the data, mainly used to store the data and some useful attributes (such as the component each datapoint belongs to).
    * `MOCKGenotype`- Class for storing the MST-derived base genotype, ranking of the links, delta value etc.
* `delta_mock.py`: The main location of the GA, where most of the DEAP code to create the GA can be found. This brings together the other code needed, mainly from `precompute.py`, `initialisation.py`, `objectives.py`, amd `operators.py`.
* `evaluation.py`: A mess with a simple function - get the number of clusters and ARIs for each individual in the final population.
* `graph_funcs.py`: A truly horrible mess of any function I ever used to make a graph. I recommend at looking at `plots.py` - this is kept as it may contain useful tricks/code for future matplotlib.
* `initialisation.py`: Functions for creating the initial population.
* `objectives.py`: Where the functions for calculating the two objectives are stored.
* `operators.py`: The crossover and mutation operators are defined here. If a Genotype class is developed, some of these may be better located within that, but for now they just work on lists and are separate. As a result of experimentation, there are a few (very similar) operators here.
* `plot_components.py`: A quick script written to plot what the components actually look like on a dataset at a chosen level of delta. For presentation and investigation purposes.
* `plots.py`: An attempt at trying to more flexibly generate graphs. There is a param dict at the bottom that can be used to define most of what you could possibly need.
* `precompute.py`: Functions for the require precomputation, such as calculating the MST.
* `run_mock.py`: The main script that runs MOCK for each of the parameter settings.
* `tests.py`: Just a simple test to compare the output of the algorithm with previously saved results. Only used when the validate command-line argument is given.
* `utils.py`: Some general functions, mainly related to command line parsing and config checking.

# Config Guide
Below is a short explanation of each of the fields for the config file. An example can be found in the validation sub-folder.

* `"exp_name"`: The name for the experiment, which will be the name for the folder.
* `"data_folder"`: The location of the folder where the data is (given as a relative path to the base directory, separated with `"/"`)
* `"data_subset"`: A string that will select a subset of the data in the folder. Leave as `null` for every file in the folder to be used.
* `"num_runs"`: The number of runs (e.g. independent seeds)
* `"num_gens"`: The number of generations
* `"num_indivs"`: The number of individuals in a population
* `"delta_sr_vals"`: Which values of delta to use, in terms of multiples of the square root (as in Mario's paper)
* `"delta_raw_vals"`: Raw delta values to use (between 0 and 100)
* `"mut_method"`: Which mutation method to use (choice of "original", "centroid", or "neighbour")
* `"crossover_prob"`: The probability of crossover.
* `"L"`: The neighbourhood parameter
* `"L_comp"`: The component neighbourhood parameer used in the other mutation methods (i.e. not "original")
* `"strategies"`: The strategy to use - only important for the adaptive version of MOCK.
* `"seed_file"`: For reproducible experiments, provide the name to a seed file. It is expected that this is in the `seeds` subdirectory. If none is provided, random seeds are generated and saved in the experiment folder.

# Code Overview
## Directory Structure & Saving/Loading
In the main directory, there are several key directories: `configs`, `experiments`, and `validation`.

For ease, the configs where experiments are set up are stored in the `configs` folder. At the command line, just the name of the config needs to be given, as it is assumed that it is in this folder. The config is then saved in the relevant experiment folder (discussed below). Some processing of the config occurs during the run, mainly filling in any gaps. The saved config in the experiment folder will be the corrected version, but this will not overwrite the one in the `configs` directory. This is intentional, but can change.

The `validation` folder contains a single dataset, and some previous results to ensure that the algorithm performs consistently over changes. Of course, intentional algorithmic changes will violate this validation, but the options to pass this should always remain (as it is the default working of Delta-MOCK). Unit testing is not currently implemented.

The path to the folder where the data is kept is given in the config, so this is up to the user.

During a run, a folder is created in the `experiments` subdirectory, according to the `exp_name` parameter in the config. This is the "experiment folder". For reproducibility, the config (as mentioned) is saved in this folder. The seed file is saved in the root `seeds` directory so it can be used by other experiments if desired. The results are also saved in this experiment folder.


## Processing Data
At the moment, the loading data step is quite hard-coded to the unusual format that the data comes in. A single function and flag or something may be needed when we start to deal with data from different sources, though I expect most data will just be a plain array and not have the preamble that the current datasets have.

Regardless, this is handled in the `Datapoint` as mentioned above. Generalizing this will just require creating a central function, and then having the specific processing functions that this calls as necessary. Some modification would also be needed in `run_mock.prepare_data()` to make it more general.


## Changes to Delta
When a change in delta is triggered, there are some parts of an individual that need to be adjusted. Previously, the method for this can be seen in `delta_mock.adaptive_delta_trigger()`. Here, the reduced genotypes are extended to what they should be, the new delta value is stored, the new components are determined, etc.


## Structural Changes for Varying Delta per Individual
It may be that a Genotype class is needed. Originally this was planned but with deadlines I had other things to focus on, which is now coming back to haunt me. The MOCKGenotype class was a quick compromise, but should probably now be integrated such that each individual is an instance of this class (and then wrapped by DEAP). Then the `operators.py` script would probably be better suited as instance or class methods of this genotype class.

If you just need one or two attributes available for the individual, these can be specified in the DEAP individual creator (at the top of `delta_mock.py`). If more functionality is needed (which will 99% be the case), then you'll need to just make the Genotype class and then give that to DEAP (instead of `list`, as seen in `delta_mock.py`, line 18).


### Delta-Evaluation
The `PartialClust` class is one of the main parts for the delta-evaluation part, which is there to help speed up evaluation. As discussed, this may change when delta is controlled by the individual itself. In the scenario where delta-evaluation is no longer used, it may be better to cut out this code entirely, and therefore rewrite the objective functions to a simpler (but obviously more computationally expensive) wholesale calculation of the intracluster variance and connectivity.


## Plots
I have rewritten the way that results are stored, so it's likely that 0% of the plotting functions now work. The core of many of them should still be of use, it's just loading in the data and selecting the right part that will be different. If you have issues here let me know, as this isn't something I will do in the future without a purpose (e.g. someone needing it, or a paper).