import torch
import gpytorch as gp
from tqdm import tqdm

from utils.gp_models import ApproximateGPModel

def train(
    model_name: str,
    model: ApproximateGPModel,
    likelihood: gp.likelihoods.Likelihood,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    learning_rate: float,
    training_iterations: int,
    objective_type: str,
    device: torch.device,
):
    """
    Trains a Gaussian process model using a specified objective function.

    This function performs optimization of the model parameters using the 
    Adam optimizer to minimize the chosen objective function. It supports 
    two types of objective functions: "ppgpr" (Predictive Log Likelihood) 
    and "svgp" (Variational Evidence Lower Bound). The optimization is 
    performed for a specified number of iterations.

    Args:
        model_name (str): The name of the model being trained (for logging purposes).
        model (ApproximateGPModel): The Gaussian Process model to be trained.
        likelihood (gp.likelihoods.Likelihood): The likelihood function associated with the model.
        train_x (torch.Tensor): Input training data (features) of shape (N, D), where N is the number of data points and D is the number of dimensions.
        train_y (torch.Tensor): Target training data (labels) of shape (N,).
        learning_rate (float): The learning rate for the optimizer.
        training_iterations (int): The number of iterations (epochs) to run during training.
        objective_type (str): The type of objective function used for training. Options are:
            - "ppgpr" for predictive log likelihood.
            - "svgp" for variational evidence lower bound (ELBO).
        device (torch.device): The device (CPU or GPU) on which to perform training.

    Returns:
        Tuple[ApproximateGPModel, gp.likelihoods.Likelihood]:
            - The trained Gaussian Process model.
            - The trained likelihood function.

    Raises:
        ValueError: If the provided `objective_type` is not recognized.

    Example:
        model, likelihood = train(
            model_name="GP_Model_1",
            model=my_model,
            likelihood=my_likelihood,
            train_x=my_train_x,
            train_y=my_train_y,
            learning_rate=0.01,
            training_iterations=1000,
            objective_type="ppgpr",
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
    """
    
    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()},
    ], lr=learning_rate)

    if objective_type == "ppgpr":
        mll = gp.mlls.PredictiveLogLikelihood(likelihood, model, num_data=train_y.numel()).to(device)
    elif objective_type == "svgp":
        mll = gp.mlls.VariationalELBO(likelihood, model, num_data=train_y.numel()).to(device)
    else:
        raise ValueError(f"Unknown objective type: {objective_type}")

    print("Training model: ", model_name)
    for i in tqdm(range(training_iterations)):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        if i == training_iterations - 1:
            print(f"Final {model_name} loss: {loss.item():.5f}")
        loss.backward()
        optimizer.step()

    print(f"Training {model_name} complete.")

    return model, likelihood
