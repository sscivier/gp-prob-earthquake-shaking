# Gaussian Processes for Probabilistic Estimates of Earthquake Ground Shaking: A 1-D Proof-of-Concept, in prep.

This repository is the official implementation of _Gaussian Processes for Probabilistic Estimates of Earthquake Ground Shaking: A 1-D Proof-of-Concept_, in prep.

We present a proof-of-concept workflow for probabilistic earthquake ground motion prediction that accounts for inconsistencies between existing seismic velocity models. The approach is based on the probabilistic merging of overlapping seismic velocity models using scalable Gaussian Process regression (GPR). We fit a GP to two synthetic 1-D velocity profiles simultaneously, demonstrating that the predictive uncertainty accounts for the differences between them. We then draw samples of velocity models from the predictive distribution and simulate the acoustic wave equation using each sample as input. This results in a distribution of possible peak ground displacement (PGD) scenarios, reflecting the uncertainty in our knowledge of seismic velocities in the region.

## Installation and requirements

This repository contains files with two options: either using CUDA for GPU acceleration during model training, or not. CUDA users should use files ending in `_cuda`, otherwise use files ending in `_cpu`.

This code has been tested using Python 3.10.12. Using a different Python version may result in package conflicts.

To create a virtual environment and install requirements:

**CUDA:**

Navigate to working directory. Create ```venv``` (change name by replacing ```my-venv```), and activate:
```shell
python3 -m venv my-venv
source my-env/bin/activate
```

Upgrades:
```shell
pip install -U pip setuptools wheel
```

Install PyTorch:
```shell
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Install remaining requirements:
```shell
pip install -r requirements_cuda.txt
```

**CPU:**

Navigate to working directory. Create ```venv``` (change name by replacing ```my-venv```), and activate:
```shell
python3 -m venv my-venv
source my-env/bin/activate
```

Upgrades:
```shell
pip install -U pip setuptools wheel
```

Install PyTorch:
```shell
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Install remaining requirements:
```shell
pip install -r requirements_cpu.txt
```

## Running the code

All of the code necessary to reproduce the results in the paper is contained in the Jupyter Notebooks. For CUDA users: `workflow_cuda.ipynb`. For CPU users: `workflow_cpu.ipynb`.

By default, the training code in the notebooks is commented out. If you wish to run the training code, simply uncomment it. These cells are indicated in the notebooks.

Pre-trained models are found in the `models/` directory. These are loaded by default in the notebooks.

## Contributing, questions, and issues

If you would like to contribute to this work, have questions, or encounter any issues with the code, please get in touch (link temporarily removed).

## References and Acknowledgements

This repository uses the following open-source software libraries:

- Jupyter Notebooks [[Website](https://jupyter.org/)][[GitHub](https://github.com/jupyter/notebook)][![DOI](https://zenodo.org/badge/DOI/10.3233/978-1-61499-649-1-87.svg)](https://doi.org/10.3233/978-1-61499-649-1-87)
- NumPy v2.1.0 [[Website](https://numpy.org/)[[GitHub](https://github.com/numpy/numpy)][![DOI](https://zenodo.org/badge/DOI/10.1038/s41586-020-2649-2.svg)](https://doi.org/10.1038/s41586-020-2649-2)
- SciPy v1.14.1 [[Website](https://scipy.org/)][[GitHub](https://github.com/scipy/scipy)][![DOI](https://zenodo.org/badge/DOI/10.1038/s41592-019-0686-2.svg)](https://doi.org/10.1038/s41592-019-0686-2)
- Matplotlib v3.9.2 [[Website](https://matplotlib.org/)][[GitHub](https://github.com/matplotlib/matplotlib)][![DOI](https://zenodo.org/badge/DOI/10.1109/MCSE.2007.55.svg)](https://doi.org/10.1109/MCSE.2007.55)
- PyTorch v2.4.0 [[Website](https://pytorch.org/)][[GitHub](https://github.com/pytorch/pytorch)][![DOI](https://zenodo.org/badge/DOI/10.48550/arXiv.1912.01703.svg)](https://doi.org/10.48550/arXiv.1912.01703)
- GPyTorch v1.12 [[Website](https://gpytorch.ai/)][[GitHub](https://github.com/cornellius-gp/gpytorch)][![DOI](https://zenodo.org/badge/DOI/10.48550/arXiv.1809.11165.svg)](https://doi.org/10.48550/arXiv.1809.11165)
