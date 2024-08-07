import numpy as np
import numpy.linalg as la
import scipy as sp
from pfapack import pfaffian as pf
from assemblers import kit_hamiltonian, dissipator, correlation_matrix


def compute_particle_density(mu: float, t: float, delta: float, gamma_g: float, gamma_l: float, sites: int) -> np.ndarray:
    """
    Calculates the average particle density in the NESS for each site of the chain.
    Args:
        mu: onsite potential
        t: hopping amplitude
        delta: superconducting gap
        gamma_g: rate of fermion creation
        gamma_l: rate of fermion annihilation
        sites: number of sites of the Kitaev chain

    Returns:
    The array density of length sites, where density[i] represents the average density on site i.
    """
    H = kit_hamiltonian(mu, t, delta, sites)
    D = dissipator(gamma_g, gamma_l, sites)
    C = correlation_matrix(H, D, sites)
    M = 1j*(C - np.identity(2*sites, dtype=np.complex128))

    density = np.zeros(sites, dtype=np.complex128)
    for i in range(sites):
        density[i] = 0.5 - 0.5*1j*C[2*i, 2*i + 1]

    return density


def compute_EGP(mu: float, t: float, delta: float, gamma_g: float, gamma_l: float, sites: int) -> (float, float):
    """
    Calculates the EGP for a Kitaev chain of arbitrary lenght.
    Code is based on https://gitlab.com/cond-mat-group/redfield-ssh.
    Args:
        mu: chemical potential
        t: hopping amplitude
        delta: superconducting gap
        gamma_g: rate of fermion creation
        gamma_l: rate of fermion annihilation
        sites: number of sites of the Kitaev chain

    Returns:
    U_real, the real part of U (whose phase is the EGP).
    U_imag, the imaginary part of U (whose phase is the EGP).
    """
    N = sites

    # Build the covariance matrix M, with M_ij = i*(<w_i w_j>_NESS - delta_{ij}):
    H = kit_hamiltonian(mu, t, delta, sites)
    D = dissipator(gamma_g, gamma_l, N)
    C = correlation_matrix(H, D, N)
    M = 1j*(C - np.identity(2*N, dtype=np.complex128))

    assert (M.transpose == -M).all, "Matrix M is not antisymmetric!"

    # Building matrices Ktilde and K2tilde:
    # first build all block matrices on the diagonal:
    block_list_Ktilde = []
    block_list_K2tilde = []
    for i in range(N):
        if i == N / 2 - 2:
            block_Ktilde = -1j * np.array([[0, 1], [-1, 0]])
            block_K2tilde = np.array([[0, 0], [0, 0]])
        else:
            block_Ktilde = -1j * np.array([[0, np.tan(np.pi * (i + 2) / N)], [-np.tan(np.pi * (i + 2) / N), 0]])
            block_K2tilde = -1j * np.array([[0, np.tan(np.pi * (i + 2) / N)], [-np.tan(np.pi * (i + 2) / N), 0]])

        block_list_Ktilde.append(block_Ktilde)
        block_list_K2tilde.append(block_K2tilde)

    # Now put together all the blocks.
    # Initial row:
    Ktilde = np.concatenate((block_list_Ktilde[0], np.zeros(shape=(2, 2 * (N - 1)), dtype=np.complex128)), axis=1)
    K2tilde = np.concatenate((block_list_K2tilde[0], np.zeros(shape=(2, 2 * (N - 1)), dtype=np.complex128)), axis=1)
    # Loop over remaining rows:
    for row in range(1, N):
        row_temp1 = np.concatenate((np.zeros(shape=(2, 2 * row), dtype=np.complex128), block_list_Ktilde[row],
                                    np.zeros(shape=(2, 2 * (N - 1 - row)), dtype=np.complex128)), axis=1)
        row_temp2 = np.concatenate((np.zeros(shape=(2, 2 * row), dtype=np.complex128), block_list_K2tilde[row],
                                    np.zeros(shape=(2, 2 * (N - 1 - row)), dtype=np.complex128)), axis=1)

        Ktilde = np.concatenate((Ktilde, row_temp1), axis=0)
        K2tilde = np.concatenate((K2tilde, row_temp2), axis=0)

    # Now we have all the matrices, and we simply need to evaluate the following formula:
    # U = exp(i*pi*(N+3)/2)*prod_{k!=N/2}cos(pi*(k+1)/N)*Pf(M)*(Pf(Ktilde-M^-1)-Pf(K2tilde-M^-1))

    # Calculate Omega=prod_{k!=N/2}cos(pi*(k+1)/N)
    # Note that we could speed this up by multiplying only half the elements as e.g. with N=2
    # cos(pi 2/60) = -cos(pi*(60-2)/60) =  - cos(pi*58/60) etc.
    Omega = 1
    for k in range(1, N + 1):
        if k != N / 2 - 1:
            Omega = Omega * np.cos(np.pi * (k + 1) / N)

    # Check whether M is singular:
    if np.linalg.matrix_rank(M) < 2*N:
        print('The covariance matrix M is singular! It cannot be inverted!')
        exit()

    # Build the inverse:
    Minv = np.linalg.inv(M)

    # Calculating the Pfaffian via external module:
    if abs((Ktilde - Minv + (Ktilde - Minv).T).max()) > 1e-14 or abs(
            (K2tilde - Minv + (K2tilde - Minv).T).max()) > 1e-14:
        print("Some matrices as not perfectly skew-symmetric!")
        print("Max Error:", np.amax(
            [abs((Ktilde - Minv + (Ktilde - Minv).T).max()), abs((K2tilde - Minv + (K2tilde - Minv).T).max())]))
    pf1 = pf.pfaffian(Ktilde - Minv)
    pf2 = pf.pfaffian(K2tilde - Minv)

    # Putting all together:
    U = np.exp(np.pi * 1j / 2 * (N + 3)) * Omega * (pf.pfaffian(M) * (pf1 - pf2))
    U_real = np.real(U)
    U_imag = np.imag(U)

    return U_real, U_imag
