# MA_ZITI

## Structure of repository
The repository contains scripts for running the algorithms presented in the master thesis of Hendrik Borras.
The thesis itself can be found here: https://github.com/HenniOVP/MA_ZITI/blob/main/thesis/MA_HB_2021.pdf

### Automatic tuning of performance parameters, Chapter 4
The script implementing the algorithms of Chapter \ref{chap:tune} also interacts with parts of the Chapter on Pruning. Thus the script is designed to be run within the Docker container created by FINN. The script itself can be found here: https://github.com/HenniOVP/MA_ZITI/tree/main/simd-pe-tuning

### Pruning in FINN, Chapter 5

The script developed for the training of pruned networks with the iterative L^1-norm pruning method can be found here: https://github.com/HenniOVP/MA_ZITI/tree/main/training

## Setup and runnig scripts

### Automatic tuning of performance parameters, Chapter 4
The jupyter notebook (https://github.com/HenniOVP/MA_ZITI/blob/main/simd-pe-tuning/cnv_varied_bit_and_pruning_parallel_testing-0.4b_dev.ipynb) should be run from within the FINN docker container.

Setting up the required jupyter notebook server with FINN:

* First setup FINN using the steps described here: https://finn.readthedocs.io/en/latest/getting_started.html
* Then clone Hendrik Borras's fork of the FINN repository: https://github.com/HenniOVP/finn
* Switch to the branch "feature/0.4_cutting_pruning".
* Copy the "cnv_varied_bit_and_pruning_parallel_testing-0.4b_dev.ipynb" notebook of this repository into the "notebooks" folder of the cloned FINN repository
* Run the jupyter notebook server in the feature branch from the command line: `bash ./run-docker.sh notebook`

Running the script:
* Open the script from jupyter notebook

Results from the script:
* Running the script produces ziped JSON files, which contain information about the optimization and results from running the given networks on an Ultra96V2 FPGA.

### Pruning in FINN, Chapter 5

Setting up Brevitas and PyTorch:
* Install Anaconda: https://www.anaconda.com/
* Create a conda enviroment using the brevitas_torch-1-4.yml contained in this repository. The enviroment contains the Brevitas and PyTorch version used in the thesis work. The command could look something like this: `conda env create -f brevitas_torch-1-4.yml`
* Create some result json files using the script for automaticaly tuning the performance parameters and store them on the computer for training.
* Create a folder structure relative to the location of the script, where the resultin json files will be stored. Like this: `mkdir -p finn_result_jsons/after_training/`

Running the script:
* Activate the conda enviroment: `conda activate brevitas_torch-1-4`
* Run the script and give it the path to the ziped json file from script from chapter 4. As example like this: `python Brevitas_train_pruning_from_FINN_json.py --finn_json finn_result.json.gz`

Results from the script:
* During training the script will print out log data and store the log in a sub-folder, which is created when the script starts. Additionally, when the script ends the final model will be saved to this folder.
* At the end of the training the script will save the loss, accuracy and model information to the path: `finn_result_jsons/after_training/` relative to its location.