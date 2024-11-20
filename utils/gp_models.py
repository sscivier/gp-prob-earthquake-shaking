import torch
import gpytorch as gp

# GP model for input data generation
class GPModel(gp.models.ExactGP):
    """
    Gaussian Process model for exact inference.

    Args:
        kernel (gpytorch.kernels.Kernel): The covariance kernel for the GP model.

    Attributes:
        mean_module (gpytorch.means.Mean): The mean module for the GP model.
        covar_module (gpytorch.kernels.Kernel): The covariance module for the GP model.

    """

    def __init__(
            self,
            kernel: gp.kernels.Kernel,
    ):
        """
        Initialize the GPModel.

        Args:
            kernel (gpytorch.kernels.Kernel): The covariance kernel for the GP model.

        """
        # No training data, so pass dummy tensors with shape (1, 1)
        super(GPModel, self).__init__(torch.zeros(1, 1), torch.zeros(1, 1), gp.likelihoods.GaussianLikelihood())
        self.mean_module = gp.means.ZeroMean()
        self.covar_module = kernel

    def forward(
            self,
            x: torch.Tensor,
    ):
        """
        Forward pass of the GPModel.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            gpytorch.distributions.MultivariateNormal: The output distribution.

        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gp.distributions.MultivariateNormal(mean_x, covar_x)
    

# GP model for velocity model fusion
class ApproximateGPModel(gp.models.ApproximateGP):
    """
    Approximate Gaussian Process model for variational inference.

    Args:
        inducing_points (torch.Tensor): The inducing points for the variational distribution.
        kernel (gpytorch.kernels.Kernel): The covariance kernel for the GP model.

    Attributes:
        mean_module (gpytorch.means.Mean): The mean module for the GP model.
        covar_module (gpytorch.kernels.Kernel): The covariance module for the GP model.

    """

    def __init__(
            self,
            inducing_points: torch.Tensor,
            kernel: gp.kernels.Kernel,
    ):
        """
        Initialize the ApproximateGPModel.

        Args:
            inducing_points (torch.Tensor): The inducing points for the variational distribution.
            kernel (gpytorch.kernels.Kernel): The covariance kernel for the GP model.

        """
        variational_distribution = gp.variational.CholeskyVariationalDistribution(
            inducing_points.size(-1)
        )
        variational_strategy = gp.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super().__init__(variational_strategy)
        self.mean_module = gp.means.ConstantMean()
        self.covar_module = gp.kernels.ScaleKernel(kernel)

    def forward(
            self,
            x: torch.Tensor,
    ):
        """
        Forward pass of the ApproximateGPModel.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            gpytorch.distributions.MultivariateNormal: The output distribution.

        """
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gp.distributions.MultivariateNormal(mean_x, covar_x)
    