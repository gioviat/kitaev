import numpy as np
from numpy import linalg as la
from scipy.linalg import ishermitian as herm, solve_continuous_lyapunov as lyap
import matplotlib.pyplot as plt


def SSH_hamiltonian(t, tprime, sites, PBC=False):
    """
       Constructs the SSH Hamiltonian matrix in Majorana representation.

       Parameters
       ----------
       t: float, A->B (intracell) hopping parameter.

       tprime: float, B->A (intercell) hopping parameter.

       sites: integer, number of sites (should be an even number).

       PBC: boolean, flag for periodic boundary conditions.

       Returns
       -------
       H, numpy array of shape 2*sites x 2*sites, Hamiltonian matrix

       Notes
       -----
       Final matrix shape:

                 |   0     0     0    -t     0     0     0    ... |
                 |   0     0     t     0     0     0     0    ... |
        T = i/4  |   0     -t    0     0     0     -t'   0    ... |
                 |   t     0     0     0     t'    0     0    ... |
                 |   0     0     0    -t'    0     0     0    ... |
                 |   0     0     t'    0     0     0     t    ... |
                 |   0     0     0     0     0     -t    0    ... |
                 |  ...   ...   ...   ...   ...   ...   ...   ... |


       Examples
       --------
       trivial topology: SSH_Hamiltonian(1.0, 0.5, 20)
       edge-state topology: SSH_Hamiltonian(0.5, 1.0, 20)

       """

    # Construct first upper diagonal:
    diag1 = np.zeros(shape=(2 * sites - 1), dtype=np.complex128)
    diag_noise = np.zeros(shape=(2 * sites - 1), dtype=np.complex128)
    for i in range(len(diag1)):
        if i % 4 == 1:
            diag1[i] = 1j * t / 4
        elif i % 4 == 3:
            diag1[i] = 1j * tprime / 4

    # Construct third upper diagonal
    diag3 = -diag1[1:-1]

    # add diagonals to get the upper-triangular part of the matrix:
    H = np.diag(diag1, 1) + np.diag(diag3, 3)

    # add PBC elements:
    if PBC == True:
        H[0, 2 * sites - 1] = -1j * tprime / 4
        H[1, 2 * sites - 2] = 1j * tprime / 4

    # build the Hermitian conjugate to fill the whole matrix:
    H += H.conj().T

    visualization = True
    if visualization:
        fig, ax = plt.subplots()
        ax.matshow(H.imag)
        for (i, j), z in np.ndenumerate(H.imag):
            ax.text(j, i, '{:0.3f}'.format(z), ha='center', va='center')
    plt.close()
    return H


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
    The antisymmetric square matrix H being the Kitaev Hamiltonian in the Majorana representation.
    """
    H = np.zeros([2*sites, 2*sites], dtype=np.complex128)
    jx = t - delta
    jy = t + delta

    PBC = True

    for n in range(0, sites - 1):
        H[2*n, 2*n + 1] = mu
        H[2*n + 1, 2*n] = -mu
        H[2*n + 3, 2*n] = -jx
        H[2*n, 2*n + 3] = jx
        H[2*n + 1, 2*n + 2] = -jy
        H[2*n + 2, 2*n + 1] = jy

    H[2*(sites - 1), 2*(sites - 1) + 1] = mu
    H[2*(sites - 1) + 1, 2*(sites - 1)] = -mu

    if PBC:
        H[0, 2*sites - 1] = jy
        H[2*sites - 1, 0] = -jy
        H[2*sites - 2, 1] = jx
        H[1, 2*sites - 2] = -jx

    H = (1j/4)*H

    assert (H.T == -H).all, "Majorana Hamiltonian is not antisymmetric!"

    visualization = False
    if visualization:
        fig, ax = plt.subplots()
        ax.matshow(H.imag)
        for (i, j), z in np.ndenumerate(H.imag):
            ax.text(j, i, '{:0.3f}'.format(z), ha='center', va='center')
        fig.suptitle('Kitaev Hamiltonian with $\mu=%.2f$, $t=%.2f$, $\Delta=%.2f$ and $%d$ sites' % (mu, t, delta, sites))

    return H


def dissipator(gamma_g: float, gamma_l: float, sites: int) -> np.ndarray:
    """
    Calculates the matrix M encoding dissipation in the Lindblad approximation, for baths that
    create and annihilate fermions on each site.

    Parameters
    ----------
    gamma_g: rate of fermion creation
    gamma_l: rate of fermion annihilation
    sites: number of sites of the Kitaev chain

    Returns
    -------
    The Hermitian square matrix M, describing the effect of dissipation as a result on the interaction with the
    environment.
    """

    # Create a list of the vectors describing the effect of the environment on each site in the Majorana representation
    l_mu_list = []


    for i in range(2 * sites):
        l_mu = np.zeros(2 * sites, dtype=np.complex128)
        if i % 2 == 0:
            l_mu[i] = np.sqrt(gamma_g) / 2
            l_mu[i + 1] = 1j * np.sqrt(gamma_g) / 2
        else:
            l_mu[i - 1] = np.sqrt(gamma_l) / 2
            l_mu[i] = -1j * np.sqrt(gamma_l) / 2
        l_mu_list.append(l_mu)


    # Initialize and calculate the matrix M as the sum over all the l_mu's of the Kronecker product of l_mu and l_mu^*
    M = np.zeros([2*sites, 2*sites], dtype=np.complex128)
    for k in range(2*sites):
        M += np.kron(l_mu_list[k], l_mu_list[k].conj()).reshape(2*sites, 2*sites)

    assert herm(M), 'The bath matrix is not Hermitian!'

    visualization = False
    if visualization:
        fig, ax = plt.subplots()
        ax.matshow(M.real)
        for (i, j), z in np.ndenumerate(M.real):
            ax.text(j, i, '{:0.3f}'.format(z), ha='center', va='center')
        fig.suptitle('Real part of the dissipation matrix M, with $\gamma_g=%.2f$, $\gamma_l=%.2f$' % (gamma_g, gamma_l))
        fig, ax = plt.subplots()
        ax.matshow(M.imag)
        for (i, j), z in np.ndenumerate(M.imag):
            ax.text(j, i, '{:0.3f}'.format(z), ha='center', va='center')
        fig.suptitle('Imaginary part of the dissipation matrix M, with $\gamma_g=%.2f$, $\gamma_l=%.2f$' % (gamma_g, gamma_l))
        plt.show()
        plt.close()

    return M


def correlation_matrix(H: np.ndarray, M: np.ndarray, sites: int) -> np.ndarray:
    """
    Calculates the matrix Z, solution of the continuous time Lyapunov equation, and verifies that
    all the rapidities lie away from the imaginary axis; then uses it to calculate the correlation matrix C.

    Parameters
    ----------
    H: Kitaev Hamiltonian
    M: dissipation matrix
    sites: number of sites in the system

    Returns
    -------
    The square antisymmetric correlation matrix C, containing all the information about the system in the NESS.
    """

    X = -2.*1j*H + 2.*M.real

    Z = lyap(X.T, -M.imag)
    assert (Z.T == -Z).all, "Matrix Z is not antisymmetric!"

    evals = la.eigvals(X)
    assert all(evals), "The rapidities do not lie all away from the imaginary axis!"

    C = np.identity(2*sites, dtype=np.complex128) + 4.*1j*Z
    assert (C.T == -C).all, "The correlation matrix C is not antisymmetric!"

    print('Assemblers: The rank of the covariance matrix is',
          np.linalg.matrix_rank(1j*(C - np.identity(2*sites, dtype=np.complex128))))

    visualization = False

    if visualization:
        plt.matshow(C.real)
        plt.colorbar()
        plt.title('Real part of the correlation matrix')
        plt.matshow(C.imag, cmap='inferno')
        plt.colorbar()
        plt.show()
        plt.close()

    return C
