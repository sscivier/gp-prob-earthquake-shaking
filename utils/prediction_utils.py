import torch
import numpy as np
import gpytorch as gp

from utils.gp_models import ApproximateGPModel


def predict(
    model: ApproximateGPModel,
    x: torch.Tensor,
    likelihood: gp.likelihoods.Likelihood | None = None,
    num_samples: int | None = None,
):
    """
    Predict the output of the GP model.

    Args:
        model (ApproximateGPModel): The GP model.
        x (torch.Tensor): The input tensor.
        likelihood (gpytorch.likelihoods.Likelihood): The likelihood function.
        num_samples (int): The number of samples to draw from the distribution.

    Returns:
        torch.Tensor: The mean of the output distribution.
        torch.Tensor: The standard deviation of the output distribution.
        torch.Tensor: The samples drawn from the output distribution.

    """
    
    with torch.no_grad():

        if likelihood is not None:
            predicted_distribution = likelihood(model(x))

        else:
            predicted_distribution = model(x)

    if num_samples is not None:
        samples = predicted_distribution.sample(sample_shape=torch.Size([num_samples])).numpy()
        return predicted_distribution.mean, predicted_distribution.stddev, samples
    
    else:
        return predicted_distribution.mean, predicted_distribution.stddev


def pdf_num_sigma(
    model: ApproximateGPModel,
    test_coordinates: torch.Tensor,
    test_velocities: torch.Tensor,
    likelihood: gp.likelihoods.Likelihood | None = None,
    num_sigma_clip: float = 5.,
):
    """
    Compute the probability density function (pdf) predicted by the GP model in terms of the number of standard deviations from the mean.

    Args:
        model (ApproximateGPModel): The GP model.
        test_coordinates (torch.Tensor): The test coordinates.
        test_velocities (torch.Tensor): The test velocities.
        likelihood (gpytorch.likelihoods.Likelihood): The likelihood function.
        num_sigma_clip (float): The maximum number of standard deviations to clip the pdf.

    Returns:
        np.ndarray: The pdf in terms of the number of standard deviations from the mean.

    """    
    
    prob_density = torch.zeros((test_coordinates.shape[0], test_velocities.shape[0]))
    
    for i in range(test_coordinates.shape[0]):
        mean, std = predict(model, test_coordinates[i].unsqueeze(0), likelihood)

        prob_density[i] = torch.exp(-0.5 * ((test_velocities - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))

    prob_density = prob_density.numpy().T

    # normalise the pdf
    prob_density /= np.max(prob_density, axis=0)

    # transform to number of sigma from the mean
    prob_density_num_sigma = (-2 * np.log(prob_density)) ** 0.5

    # clip the values to a maximum of num_sigma_clip
    prob_density_num_sigma = np.clip(prob_density_num_sigma, 0, num_sigma_clip)

    return prob_density_num_sigma