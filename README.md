# Deep ARNowcasting

This project proposes a flexible framework for the training of multi-horizon autoregressive deep learning (DL) models.

The structure of the repository is the following :
  * `configs/` contains all the configurations related to the training of models,
  * `nowproject/` contains all the modules and building blocks of the project
  * `scripts/` contains different useful scripts including computing benchmarks and training models


## 1. Install required libraries

Create a conda environment using the provided `environment.yml` file:
```shell
conda env create -f environment.yml
```

The environment file does not include [PyTorch](https://pytorch.org/get-started/locally/), to accomodate for the version that your setup is compatible with. Almost all of the code runs with torch 1.9, but in order to use the Optical Flow component, you have to have the latest version.

This project made use of 3 additonal packages that can be retrieved with the following commands:
```shell
git clone git@github.com:ghiggi/xverif.git
cd xverif
git checkout categorical_scores

git clone git@github.com:ghiggi/xforecasting.git
cd xforecasting
git checkout nowcasting_changes

git clone git@github.com:ghiggi/xscaler.git
```

To use those packages, we need to add their paths to the `bashrc`:
```shell
# add xscaler, xforecasting and xscaler to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/home/haddad/xscaler"
export PYTHONPATH="${PYTHONPATH}:/home/haddad/xforecasting"
export PYTHONPATH="${PYTHONPATH}:/home/haddad/xverif"
export PYTHONPATH="${PYTHONPATH}:/home/haddad/nowproject/"
```
Then apply the command ```source ~/.bashrc```.

## 2. Pipeline

The pipeline's components can be found under `nowproject/`:
  * `training.py`: contains the AutoregressiveTraining function.
  * `predictions.py`: contains the AutoregressivePredictions function.
  * `dataloader.py`: contains all the classes and functions related to the autoregressive dataset and dataloader used in this project.
  * `loss.py`: contains the MSE, LogCosh and FSS losses implemented in this project.
  * `scalers.py`: contains the different data scalers used in this project. `utils/scalers_modules` contains the scalers along with the parameters we employed.
  * `architectures.py`: contains the different 3D architectures implemented in this project. The layers used to build those models can be found under `dl_models/`, in the files `layers_3d.py`, `layers_res_conv.py` and `layers_optical_flow.py`.
  * `architectures_2d.py`: contains archived 2D architectures that were considered at an earlier stage in the project.
  * `plot_precip.py` and `plot_map.py` under `utils/`: contain code to plot data on maps, and in our case precipitation.
  * `plot_skills.py` under `utils/`: contain code to plot skills of benchmarks and models.

