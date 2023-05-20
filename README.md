Supplementary material for the Neurips2023 submission entitled

MMGP: a Mesh Morphing Gaussian Process-based machine learning method for regression of physical problems under nonparametrized geometrical variability

This folder contains datasets corresponding to the 2D solid mechanics dataset, see Section 4.2.

WARNING: to satisfy the 200MB upload limit, the node coordinates and output fields are provided as np.float16 arrays, while the results of the paper have been computed with np.float32 objects. Notice that you need approximately 2GB of hard drive space to run the simulations.

Folder content:

- utils.py: contains the snapshotPOD and mesh morphing function not available in the dependencies.

- pretreatData.py: morphs the meshes of the datasets and computes the FE interpolation for the spatial coordinate and output fields
- train.py: trains the GPs
- predictScalars.py: inference stage for the fields output (training and testing sets)
- plotScalars.py: plotting of bissect graph loading characteristics
- predictFields.py: inference stage for the scalars output (training and testing sets)
- plotFields.py: plotting of output field predictions, variance, 0.025 and 0.975 quantiles, references and relative errors in Xdmf format
- computeAndPlotHorsParamFields.py: prediction of scalar and field outputs for the out-of-distribution shapes ellipsoid and wedge, with plotting of the fields

- data/datasets.tar.xz: compressed file containing the (input,output) for the training and testing sets, as well as the two out-of-distribution cases (ellipsoid and wedge)
- env.yml: description of the dependencies of the code

To construct a compatible python environment, one can use a "conda create" with the env.yml file.

To reproduce the example of this repo, run the following commands

- tar -xf data/datasets.tar.xz
- python pretreatData.py
- python train.py
- python predictFields.py
- python plotFields.py
- python predictScalars.py
- python plotScalars.py
- python computeAndPlotHorsParamFields.py

Alternatively, the previous workflow can be executed with python run.py

We refer to the comments in these files for a description of the actions performed by these commands.
We refer to computeAndPlotHorsParamFields.py for an exemple of a complete inference workflow.
