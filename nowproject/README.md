#  Pipeline components

This folder contains the different pipeline components. At its root, you can find :
  * `training.py`: contains the AutoregressiveTraining function.
  * `predictions.py`: contains the AutoregressivePredictions function.
  * `dataloader.py`: contains all the classes and functions related to the autoregressive dataset and dataloader used in this project.
  * `loss.py`: contains the MSE, LogCosh and FSS losses implemented in this project.
  * `scalers.py`: contains the different data scalers used in this project. 
  * `architectures.py`: contains the different 3D architectures implemented in this project. The layers used to build those models can be found under `dl_models/`, in the files `layers_3d.py`, `layers_res_conv.py` and `layers_optical_flow.py`.
  * `config.py`: contains functions to process the config files passed to the pipeline.

Other components of the pipeline can be found under :
* `verification/`: contains the files related to the verification routines. 
* `visualization/`: contains the files related to the plotting functions.
* `models/`: contains the different layers and functions that compose our deep learning architectures.
* `data/`: features all the functions related to the construction of the dataset, the precipitation patch extraction and the scaling of the data.