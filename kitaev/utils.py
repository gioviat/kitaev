import numpy as np


def fermi_dirac(eps: float, beta: float, mu: float) -> float:
    """
    Return the Fermi-Dirac distribution.
    Args:
        eps: energy
        beta: inverse temperature
        mu: chemical potential

    Returns:
    The value of the Fermi-Dirac distribution for the considered parameters.
    """
    fd = 1.0 / (np.exp(beta * (eps - mu)) + 1.0)

    return fd


def bath_spectral_func(lambd: float, omega: float,
                       beta1: float, mu1: float, beta2: float, mu2: float, N: int) -> np.ndarray:
    """
    Constructs the spectral correlation function for Markovian baths in the Redfield approximation.
    Args:
        lambd: system-bath coupling constant
        omega: frequency
        beta1: inverse temperature of the first bath
        mu1: chemical potential of the first bath
        beta2: inverse temperature of the second bath
        mu2: chemical potential of the second bath
        N: number of sites

    Returns:
    The bath spectral function in Majorana representation.
    """
    # Initialize empty bath spectral matrix:
    gamma = np.zeros(shape=(2 * N, 2 * N), dtype=np.complex128)

    # Define the entries of the bath spectral function:
    A1 = fermi_dirac(omega, beta1, mu1) + (1 - fermi_dirac(-omega, beta1, mu1))
    A2 = fermi_dirac(omega, beta2, mu2) + (1 - fermi_dirac(-omega, beta2, mu2))
    B1 = fermi_dirac(omega, beta1, mu1) - (1 - fermi_dirac(-omega, beta1, mu1))
    B2 = fermi_dirac(omega, beta2, mu2) - (1 - fermi_dirac(-omega, beta2, mu2))

    # Populate the entries of the matrix:
    for i in range(N):
        if i % 2 == 0:
            gamma[2 * i, 2 * i] = A1
            gamma[2 * i + 1, 2 * i + 1] = A1
            gamma[2 * i, 2 * i + 1] = -1j * B1
            gamma[2 * i + 1, 2 * i] = 1j * B1
        if i % 2 == 1:
            gamma[2 * i, 2 * i] = A2
            gamma[2 * i + 1, 2 * i + 1] = A2
            gamma[2 * i, 2 * i + 1] = -1j * B2
            gamma[2 * i + 1, 2 * i] = 1j * B2

    gamma = lambd ** 2 * gamma

    return gamma
