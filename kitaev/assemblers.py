import numpy as np
from numpy import linalg as la
import matplotlib as mp
import matplotlib.pyplot as plt


def kit_hamiltonian(mu: float, t: float, delta: float, sites: int) -> np.ndarray:
    """
    This function calculates the Kitaev Hamiltonian in the Majorana basis
    :param mu: onsite potential
    :param t: hopping amplitude
    :param delta: superconducting gap
    :param sites: number of sites
    :return: Kitaev Hamiltonian
    """

    h = np.zeros([2*sites, 2*sites], dtype=complex)
    jx = t - delta
    jy = t + delta

    for n in range(0, sites - 1):
        h[2 * n, 2 * n + 1] = mu
        h[2 * n + 1, 2 * n] = -mu
        h[2 * n + 3, 2 * n] = jy
        h[2 * n, 2 * n + 3] = -jy
        h[2 * n + 1, 2 * n + 2] = jx
        h[2 * n + 2, 2 * n + 1] = -jx

    h[2 * (sites - 1), 2 * (sites - 1) + 1] = mu
    h[2 * (sites - 1) + 1, 2 * (sites - 1)] = -mu

    h = (1j / 4) * h

    assert np.linalg.norm(h + h.T) < 10 ** (-10), "Majorana Hamiltonian is not antisymmetric!"

    return h


def bath_operators(gamma: float, sites: int) -> np.ndarray:
    """
    This function calculates the Lindbladian jump operators in the Majorana basis,
    in the case of a one-dimensional chain connected to Markovian baths with
    local dissipation.
    :param gamma: rate of fermions creation/annihilation
    :param sites: number of sites 
    :return: matrix of jump operators
    """

    l = np.zeros([sites - 1, 2 * sites], dtype=complex)

    for n in range(sites - 1):
        l[n, 2 * n] = gamma
        l[n, 2 * n + 3] = 1j * gamma

    return l


def dissipator(h: np.ndarray, l: np.ndarray, sites: int) -> None:
    """
    This function calculates the matrices M and Z, calculates
    the eigenvalues of -Z (corresponding to the eigenvalues of
    the Lindbladian) and plots them.
    :param h: 
    :param l: 
    :param sites: 
    :return: 
    """

    m = np.dot(l.transpose(), l.conj())
    z = h + 1j*m.real

    eigenvalues = la.eigvals(-z)

    idx = eigenvalues.argsort()[::-1]
    eigenvalues = np.flip(eigenvalues[idx])

    plt.scatter(np.arange(len(eigenvalues)), eigenvalues.real)
    plt.title('Real Energy Spectrum of Chain with {} Sites'.format(sites))
    plt.ylim([-1.5, 1.5])
    plt.show()

    plt.scatter(np.arange(len(eigenvalues)), eigenvalues.imag)
    plt.title('Imaginary Energy Spectrum of Chain with {} Sites'.format(sites))
    plt.ylim([-0.05, 0])
    plt.show()

