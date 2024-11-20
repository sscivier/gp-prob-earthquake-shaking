import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt


def set_plotting_hyperparameters():
    
    # set font sizes
    plt.rcParams['axes.titlesize'] = 18
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    plt.rcParams['legend.fontsize'] = 14

    # set linewidths and tick linewidths
    plt.rcParams['axes.linewidth'] = 2
    plt.rcParams['xtick.major.width'] = 2
    plt.rcParams['ytick.major.width'] = 2

    # increase tick length
    plt.rcParams['xtick.major.size'] = 8
    plt.rcParams['ytick.major.size'] = 8

    # increase plot line widths
    plt.rcParams['lines.linewidth'] = 2

    # increase marker sizes
    plt.rcParams['lines.markersize'] = 8


def truncate_colormap(
        cmap: colors.Colormap,
        min_val: float = 0.0,
        max_val: float = 1.0,
        n: int = 256,
):
    """
    Truncate a colormap to a specified range.

    Args:
        cmap (matplotlib.colors.Colormap): The colormap to be truncated.
        min_val (float, optional): The minimum value of the colormap range. Defaults to 0.0.
        max_val (float, optional): The maximum value of the colormap range. Defaults to 1.0.
        n (int, optional): The number of colors in the truncated colormap. Defaults to 256.

    Returns:
        matplotlib.colors.Colormap: The truncated colormap.

    Raises:
        None

    Examples:
        # Truncate the 'viridis' colormap to the range [0.2, 0.8] with 128 colors
        new_cmap = truncate_colormap(plt.cm.viridis, min_val=0.2, max_val=0.8, n=128)
    """
    new_cmap = colors.LinearSegmentedColormap.from_list(
        f"trunc({cmap.name},{min_val:.2f},{max_val:.2f})",
        cmap(np.linspace(min_val, max_val, n))
    )
    return new_cmap