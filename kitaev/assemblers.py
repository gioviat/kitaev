import numpy as np
from numpy import linalg as la
from scipy.linalg import ishermitian as herm, solve_continuous_lyapunov as lyap
import matplotlib.pyplot as plt


def kit_hamiltonian(mu: float, t: float, delta: float, sites: int) -> np.ndarray:
    """
    Calculates the Kitaev Hamiltonian in the Majorana representation.

    Parameters
    ----------
    mu: onsite potential
    t: hopping amplitude
    delta: superconducting gap
    sites: number of sites

    Returns
    -------
    A square matrix being the Kitaev Hamiltonian in the Majorana representation.
    """
    h = np.zeros([2*sites, 2*sites], dtype=np.complex128)
    jx = t - delta
    jy = t + delta

    PBC = True

    for n in range(0, sites - 1):
        h[2*n, 2*n + 1] = mu
        h[2*n + 1, 2*n] = -mu
        h[2*n + 3, 2*n] = jx
        h[2*n, 2*n + 3] = -jx
        h[2*n + 1, 2*n + 2] = -jy
        h[2*n + 2, 2*n + 1] = jy

    h[2*(sites - 1), 2*(sites - 1) + 1] = mu
    h[2*(sites - 1) + 1, 2*(sites - 1)] = -mu

    if PBC:
        h[0, 2*sites - 1] = jy
        h[2*sites - 1, 0] = -jy
        h[2*sites - 2, 1] = jx
        h[2, 2*sites - 2] = -jx

    h = (1j/4)*h

    assert (h.transpose == -h).all, "Majorana Hamiltonian is not antisymmetric!"

    return h


def bath_operators(gamma_g: float, gamma_l: float, sites: int) -> np.ndarray:
    """
    Calculates the matrix l encoding dissipation in the Lindblad approximation, for baths that
    create and annihilate fermions at the same rate (i.e. the jump operators are Hermitian).

    Parameters
    ----------
    gamma: rate of creation/annihilation of fermions
    sites: number of sites of the Kitaev chain

    Returns
    -------
    The bath operators in Majorana representation
    """

    L = np.zeros([2*sites, 2*sites], dtype=np.complex128)

    for i in range(2*sites):
        if i % 2 == 0:
            L[i, i] = np.sqrt(gamma_g)/2
            L[i + 1, i] = 1j*np.sqrt(gamma_g)/2
        else:
            L[i - 1, i] = np.sqrt(gamma_l)/2
            L[i, i] = -1j*np.sqrt(gamma_l)/2

    plt.matshow(L.real)
    plt.colorbar()
    plt.title('Real part of L')
    plt.matshow(L.imag)
    plt.colorbar()
    plt.title('Imaginary part of L')
    plt.show()
    plt.close()

    return l


def dissipator(h: np.ndarray, L: np.ndarray, sites: int) -> np.ndarray:
    """
    Calculates the matrix z, solution of the continuous time Lyapunov equation, and verifies that
    all the rapidities lie away from the imaginary axis.

    Parameters
    ----------
    h: Kitaev Hamiltonian
    l: bath operators

    Returns
    -------
    The square matrix z, whose eigenvalues are the rapidities for the system.
    """

    m = np.zeros([2*sites, 2*sites], dtype=np.complex128)

    for i in range(2*sites):
        for j in range(2*sites):
            for k in range(2*sites):
                m[i, j] += L[k, i]*L[k, j].conj()

    assert herm(m), 'The bath matrix is not Hermitian!'

    plt.matshow(m.real)
    plt.colorbar()
    plt.title('Real part of M')
    plt.matshow(m.imag)
    plt.colorbar()
    plt.title('Imaginary part of M')
    plt.show()
    plt.close()

    x = -2*1j*h + 2*m.real

    z = lyap(x, m.imag)
    assert (z.transpose == -z).all, "Matrix Z is not antisymmetric!"

    evals = la.eigvals(x)
    assert all(evals), 'The rapidities do not lie all away from the imaginary axis!'


    return z


def correlation_matrix(z: np.ndarray, sites: int) -> np.ndarray:

    c = np.identity(2*sites, dtype=np.complex128) + 4*1j*z
    assert (c.transpose == -z).all, "Correlation matrix is not antisymmetric!"

    return c
