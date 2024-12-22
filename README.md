# Gaussian Processes for Probabilistic Estimates of Earthquake Ground Shaking: A 1-D Proof-of-Concept, accepted in ML4PS Workshop @ NeurIPS 2024.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/sscivier/gp-prob-earthquake-shaking/HEAD?urlpath=tree)
[![DOI](https://zenodo.org/badge/848923377.svg)](https://doi.org/10.5281/zenodo.14246055)

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Citing this work](#citing-this-work)
- [Contributing, questions, and issues](#contributing-questions-and-issues)
- [License](#license)
- [References and Acknowledgements](#references-and-acknowledgements)

## Introduction
This repository is the official implementation of _Gaussian Processes for Probabilistic Estimates of Earthquake Ground Shaking: A 1-D Proof-of-Concept_, accepted in [ML4PS Workshop](https://ml4physicalsciences.github.io/2024/) @ NeurIPS 2024.

We present a proof-of-concept workflow for probabilistic earthquake ground motion prediction that accounts for inconsistencies between existing seismic velocity models. The approach is based on the probabilistic merging of overlapping seismic velocity models using scalable Gaussian Process (GP) regression. We fit a GP to two synthetic 1-D velocity profiles simultaneously, demonstrating that the predictive uncertainty accounts for the differences between them. We then draw samples of velocity models from the predictive distribution and simulate the acoustic wave equation using each sample as input. This results in a distribution of possible peak ground displacement (PGD) scenarios, reflecting the degree of knowledge of seismic velocities in the region.

## Installation
You can use Binder to run the notebook online without any installation. Click the Binder badge at the top of this README to launch the notebooks in your browser. It is recommended **not** to run model training on Binder, as it may take much longer than on your computer.

If you wish to run the code on your computer, the installation instructions have two options: either for using CUDA for GPU acceleration during model training, or not. CUDA users should follow the **CUDA** instructions, and CPU users should follow the **CPU** instructions.

**This code has been tested using Python 3.10.12. Using a different Python version may result in package conflicts, or installation errors. If ```pip``` cannot find certain packages during installation, then ensure you are running Python 3.10.12 in your virtual environment, and retry the installation.**

Create a virtual environment:

Navigate to working directory. Create ```venv``` (change name by replacing ```my-venv```), and activate:
```shell
python3 -m venv my-venv
source my-env/bin/activate
```

Clone the repository:

```shell
git clone git@github.com:sscivier/gp-prob-earthquake-shaking.git
```

**CUDA:**

Install requirements:
```shell
pip install -r requirements_cuda.txt
```

**CPU:**

Install requirements:
```shell
pip install -r requirements.txt
```

## Usage
All of the code necessary to reproduce the results in the paper is contained in the Jupyter Notebook, `workflow.ipynb`.

By default, the training code in the notebook is commented out. If you wish to run the training code, simply uncomment it. This cell is indicated in the notebook.

Pre-trained models are found in the `trained_models/` directory. These are loaded by default in the notebook.

## Citing this work
This repository includes a `CITATION.cff` file containing the software citation. Please cite the software using this information. GitHub automatically generates APA and BibTeX citations that you can use.

Associated paper citation coming soon ...

## Contributing, questions, and issues
If you have any suggestions, improvements, questions, or comments - please create an issue, submit a pull request, or [get in touch](mailto:sam.scivier@earth.ox.ac.uk).

## License
This project is licensed under the Apache-2.0 License. See the [LICENSE](LICENSE) file for details.

## References and Acknowledgements

This repository uses the following open-source software libraries:

- Jupyter Notebooks [[Website](https://jupyter.org/)][[GitHub](https://github.com/jupyter/notebook)][![DOI](https://zenodo.org/badge/DOI/10.3233/978-1-61499-649-1-87.svg)](https://doi.org/10.3233/978-1-61499-649-1-87)
- Binder [[Website](https://mybinder.org/)][[GitHub](https://github.com/jupyterhub/binderhub)][![DOI](https://zenodo.org/badge/DOI/10.25080/Majora-4af1f417-011.svg)](https://doi.org/10.25080/Majora-4af1f417-011)
- NumPy v2.1.3 [[Website](https://numpy.org/)][[GitHub](https://github.com/numpy/numpy)][![DOI](https://zenodo.org/badge/DOI/10.1038/s41586-020-2649-2.svg)](https://doi.org/10.1038/s41586-020-2649-2)
- SciPy v1.14.1 [[Website](https://scipy.org/)][[GitHub](https://github.com/scipy/scipy)][![DOI](https://zenodo.org/badge/DOI/10.1038/s41592-019-0686-2.svg)](https://doi.org/10.1038/s41592-019-0686-2)
- Matplotlib v3.9.2 [[Website](https://matplotlib.org/)][[GitHub](https://github.com/matplotlib/matplotlib)][![DOI](https://zenodo.org/badge/DOI/10.1109/MCSE.2007.55.svg)](https://doi.org/10.1109/MCSE.2007.55)
- PyTorch v2.5.1 [[Website](https://pytorch.org/)][[GitHub](https://github.com/pytorch/pytorch)][![DOI](https://zenodo.org/badge/DOI/10.48550/arXiv.1912.01703.svg)](https://doi.org/10.48550/arXiv.1912.01703)
- GPyTorch v1.13 [[Website](https://gpytorch.ai/)][[GitHub](https://github.com/cornellius-gp/gpytorch)][![DOI](https://zenodo.org/badge/DOI/10.48550/arXiv.1809.11165.svg)](https://doi.org/10.48550/arXiv.1809.11165)

Sam A. Scivier is funded by a UKRI NERC DTP Award (NE/S007474/1) and gratefully acknowledges their support.
