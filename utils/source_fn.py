import numpy as np


def ricker_wavelet(
    time: np.ndarray,
    center_frequency: float,
    time_shift: float = 0.0,
):
    """
    Generates a Ricker wavelet, also known as the "Mexican hat" wavelet.

    The Ricker wavelet is a symmetric waveform commonly used in geophysics
    to model seismic sources. It integrates to zero and has a defined peak frequency.

    Args:
        time (np.ndarray): Time vector (seconds).
        center_frequency (float): Central frequency of the wavelet (Hz).
        time_shift (float): Time at which the wavelet is centered (seconds).

    Returns:
        np.ndarray: Ricker wavelet values at each time step.
    """
    
    term = np.pi * center_frequency * (time - time_shift)
    return (1.0 - 2.0 * term ** 2) * np.exp(-term ** 2)