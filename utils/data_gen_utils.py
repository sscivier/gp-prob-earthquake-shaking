import torch
import numpy as np
import gpytorch as gp
from typing import List

from utils.gp_models import GPModel


def set_seed(seed):
    """
    Set the seed for random number generation in PyTorch and NumPy.

    This function ensures reproducibility by setting the seed for both
    PyTorch and NumPy random number generators.

    Parameters:
    seed (int): The seed value to set for the random number generators.

    Returns:
    None
    """

    torch.manual_seed(seed)
    np.random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def sample_gp_prior(
        kernel: gp.kernels.Kernel,
        lengthscale: float,
        n_samples: int = 1,
        x: torch.Tensor = None,
        x_samples: List[torch.Tensor] = None,
):
    """
    Generate samples from a Gaussian Process (GP) prior.

    Parameters:
    - kernel (gpytorch.kernels.Kernel): The kernel function used in the GP model.
    - lengthscale (float or List[float]): The lengthscale(s) of the kernel function. If a single value is provided, it will be used for all samples. If a list is provided, the lengthscale should match the number of samples.
    - n_samples (int): The number of samples to generate from the GP prior. Default is 1.
    - x (torch.Tensor): The input tensor for the GP model. Either `x` or `x_samples` must be specified.
    - x_samples (List[torch.Tensor]): The input tensors for each sample. If not specified, `x` will be used for all samples. The length of `x_samples` should match the number of samples.

    Returns:
    - samples (torch.Tensor): The generated samples from the GP prior.
    
    """
    ...
    if isinstance(lengthscale, (int, float)):
        lengthscale = [lengthscale] * n_samples
    elif len(lengthscale) != n_samples:
        raise ValueError("Number of length scales should match the number of samples")

    if x_samples is None:
        if x is None:
            raise ValueError("Either x or x_samples must be specified")
        x_samples = [x] * n_samples
    elif len(x_samples) != n_samples:
        raise ValueError("Number of x-arrays should match the number of samples")

    samples = torch.Tensor([])
    for i in range(n_samples):
        # Set the GP model with the custom kernel and length scale
        kernel.lengthscale = torch.tensor([lengthscale[i]])
        model = GPModel(kernel)

        # Set the model in eval mode
        model.eval()

        # Sample from the GP model
        with torch.no_grad():
            sample = model(x_samples[i]).rsample()
            samples = torch.cat([samples, sample.unsqueeze(0)], dim=0)

    return samples


def normalise(
        data: torch.Tensor,
        n: float,
        m: float,
):
    """
    Normalizes the given data to a specified range.

    Args:
        data (torch.Tensor): The input data to be normalized.
        n (float): The lower bound of the desired range.
        m (float): The upper bound of the desired range.

    Returns:
        torch.Tensor: The normalized data.

    """

    data = (data - torch.min(data)) / (torch.max(data) - torch.min(data))
    data = data * (m - n) + n

    return data
