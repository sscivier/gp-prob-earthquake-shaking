# Gaussian Processes for Probabilistic Estimates of Earthquake Ground Shaking: A 1-D Case Study

This repository is the official implementation of _Gaussian Processes for Probabilistic Estimates of Earthquake Ground Shaking: A 1-D Case Study_, in prep.

## Requirements

This repository contains files with two options: either using CUDA for GPU acceleration during model training, or not. CUDA users should use files ending in `_cuda`, otherwise use files ending in `_cpu`.

This package has been tested using Python 3.10.12. Using a different Python version may result in package conflicts.

To install requirements:

CUDA:
```
pip install -r requirements_cuda.txt
```

CPU:
```
pip install -r requirements_cpu.txt
```

## Running the code

All of the code necessary to reproduce the results in the paper is contained in the Jupyter Notebooks. For CUDA users: `workflow_cuda.ipynb`. For CPU users: `workflow_cpu.ipynb```.

By default, the training code in the notebooks is commented out. If you wish to run the training code, simply uncomment it. These cells are indicated in the notebooks.

Pre-trained models are found in the `models/` directory. These are loaded by default in the notebooks.

## Contributing, questions, and issues

If you would like to contribute to this work, have questions, or encounter any issues with the code, please [get in touch](mailto:sam.scivier@earth.ox.ac.uk).
