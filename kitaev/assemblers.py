import numpy as np


def kit_hamiltonian(mu: float, t: float, delta: float, sites: int) -> np.ndarray:
    """
    This function calculates the Kitaev Hamiltonian in the Majorana basis
    :param mu: onsite potential
    :param t: hopping amplitude
    :param delta: superconducting gap
    :param sites: number of sites
    :return: Kitaev Hamiltonian
    """

    h = np.zeros([2*sites, 2*sites])
    gammaminus = t - delta
    gammaplus = t + delta

    for n in range(sites - 1):
        h[2*n, 2*n + 1] = -gammaplus
        h[2*n + 1, 2*n] = gammaplus
        h[2*n - 1, 2*n + 2] = gammaminus
        h[2*n + 2, 2*n - 1] = gammaminus
        h[2*n - 1, 2*n] = mu
        h[2*n, 2*n - 1] = mu

    h[2*(sites - 1) - 1, 2*(sites - 1)] = mu
    h[2*(sites - 1), 2*(sites - 1) - 1] = -mu
    h = (1j/4)*h

    return h


def bath_operators(loss: float, gain: float, sites: int) -> np.ndarray:
    """
    This function calculates the Lindbladian jump operators in the Majorana basis,
    in the case of a one-dimensional chain connected to Markovian baths with
    local dissipation.
    :param loss: rate of fermions annihilation
    :param gain: rate of fermions creation
    :param sites: number of sites 
    :return: matrix of jump operators
    """

    l = np.zeros([sites, 2*sites])

    for n in range(sites):
        l[n, 2*n - 1] = (np.sqrt(loss) + np.sqrt(gain))/2
        l[n, 2*n] = 1j*(-np.sqrt(loss) + np.sqrt(gain))/2

    return l


def dissipator(h: np.ndarray, l: np.ndarray, sites: int) -> np.ndarray:
    """
    This function calculates the matrices A, M and V
    :param h: 
    :param l: 
    :param sites: 
    :return: 
    """
