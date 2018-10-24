# Delta-MOCK in Python
This is Python code for Delta-MOCK, including Adaptive Delta-MOCK, used in the paper *"Towards an Adaptive Encoding for Evolutionary Data Clustering"*<sup>1</sup>.

<sup>1</sup>Cameron Shand, Richard Allmendinger, Julia Handl, and John Keane. 2018. Towards an Adaptive Encoding for Evolutionary Data Clustering. In GECCO’18: Genetic and Evolutionary Computation Conference, July 15–19, 2018, Kyoto, Japan. ACM, New York, NY, USA, 8 pages. https://doi.org/10.1145/3205455.3205506

For the original Delta-MOCK in high performance C++, please see Mario Garza-Fabre's code [here](https://github.com/garzafabre/Delta-MOCK). This repo is intended for easy development, extensions, and, simple application of Delta-MOCK.

If you are having issues using this code, please don't hesitate to contact the repo owner (cameron.shand (at) manchester.ac.uk). 

## Requirements
A `setup.py` is on the to-do list to enforce requirements, but some key requirements will be noted here:
* Python 3.6+ 
* DEAP 1.2
* python-igraph (best way is to install through conda-forge)
* Usual python stack (numpy, scipy etc.)

## Brief Intro
A variety of command line arguments are available (full definition can be found in `utils.py`).