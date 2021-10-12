# Delta-MOCK in Python
This is Python code for Delta-MOCK, including Adaptive Delta-MOCK, used in the paper *"Towards an Adaptive Encoding for Evolutionary Data Clustering"*<sup>1</sup>.

<sup>1</sup>Cameron Shand, Richard Allmendinger, Julia Handl, and John Keane. 2018. Towards an Adaptive Encoding for Evolutionary Data Clustering. In GECCO’18: Genetic and Evolutionary Computation Conference, July 15–19, 2018, Kyoto, Japan. ACM, New York, NY, USA, 8 pages. https://doi.org/10.1145/3205455.3205506

For the original Delta-MOCK in high performance C++, please see Mario Garza-Fabre's code [here](https://github.com/garzafabre/Delta-MOCK). This repo is intended for easy development, extensions, and, simple application of Delta-MOCK.

If you are having issues using this code, please [raise an issue](https://github.com/sea-shunned/PyMOCK/issues/new). 

## Requirements
The key requirements are:
* Python 3.6+ 
* DEAP 1.2 (probably best to pip install from their github)
* python-igraph (best way is to install through conda-forge)
* Usual python stack (numpy, scipy etc.)

If you are on Linux, a conda environment has been provided. See the section below for install instructions.

## Quick Start
If it is available, install the conda environment, run `conda env create -f mock_env.yml`.

To ensure that your setup works, run `python run_mock.py -v`. If no errors occur, success!

To then run experiments, you need to create a config file, save it in the `configs` sub-folder, and then run `python run_mock.py -c configname.json`.

## User Guide
The full user guide can be found [here](https://github.com/sea-shunned/PyMOCK/blob/master/user_guide.md), which explains some aspects of the code and the different files.
