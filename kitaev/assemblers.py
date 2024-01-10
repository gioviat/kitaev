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

    for n in range(sites - 1):
        h[2*n, 2*n + 1] = mu
        h[2*n + 1, 2*n] = -mu
        h[2*n - 1, 2*n + 2] = jx
        h[2*n + 2, 2*n - 1] = -jx
        h[2*n - 1, 2*n] = -jy
        h[2*n, 2*n - 1] = jy

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

    l = np.zeros([sites, 2*sites], dtype=complex)

    for n in range(sites):
        l[n, 2*n - 1] = (np.sqrt(loss) + np.sqrt(gain))/2
        l[n, 2*n] = 1j*(-np.sqrt(loss) + np.sqrt(gain))/2

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

    m = np.zeros([2*sites, 2*sites], dtype=complex)
    z = np.zeros([2*sites, 2*sites], dtype=complex)

    m = np.dot(l.transpose(), l.conj())
    z = h + 1j*m.real

    eigenvalues = la.eigvals(-z)

    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]

    plt.scatter(np.arange(len(eigenvalues)), eigenvalues.real)
    plt.title('Real Energy Spectrum of Chain with {} Sites'.format(sites))
    plt.show()

    plt.scatter(np.arange(len(eigenvalues)), eigenvalues.imag)
    plt.title('Imaginary Energy Spectrum of Chain with {} Sites'.format(sites))
    plt.show()

