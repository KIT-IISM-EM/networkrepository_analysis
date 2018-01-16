# networkrepository_analysis
Several analysis scripts that were used to analyze data from 
networkrepository.com for graph automorphisms

## Install
No installation is needed but you must assure that all required packages are
available on yout system. `python setup.py --requires` shows the packages.
Most of them can be obtained using standard Python mechanisms for packages.
[pysaucy](https://github.com/FabianBall/pysaucy) and 
[pycggcrg](https://github.com/FabianBall/pycggcrg) need to be obtained 
separately, please follow their installation instructions.

All scripts were tested using Python 2.7, however, they should be compatible
with Python 3 as well.

## Usage
The purpose of those scripts is to download network/graph data, investigate this 
data with a focus on graph automorphisms, and  analyze the results.
The scripts must be executed in a given order:
1. `networkrepository_download_datasets.py`: Actually download data sets
2. `networkrepository_download_stats.py`: Download an overview of all data sets
available for later analysis
3. `networkrepository_run_analysis.py`: Execute the analysis of the data sets
4. `networkrepository_clean_dups.py`: (Optional) Interactively clean the
analysis results for duplicates in the data
5. `networkrepository_compute_stats.py`: Compute statistics on the obtained 
results

Each script file is executed calling `python <script name>.py`, most of them
require additional information as command line parameters. For each script,
the `--help` option is available.

## References
> Ball, Fabian, and Andreas Geyer-Schulz. 2018. “How Symmetric Are Real-World Graphs? A Large-Scale Study.” Symmetry 10 (1):17. https://doi.org/10.3390/sym10010029.

