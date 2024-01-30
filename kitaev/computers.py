import numpy as np
from pfapack import pfaffian as pf
from assemblers import kit_hamiltonian, bath_operators, dissipator, correlation_matrix


def compute_EGP(mu, t, delta, gamma, sites):

    N = sites

    # Build the correlation matrix M, with M_ij = i*(<w_i w_j>_NESS - delta_{ij}):
    h = kit_hamiltonian(mu, t, delta, N)
    l = bath_operators(gamma, N)
    z = dissipator(h, l)
    M = 1j*correlation_matrix(z, N) - np.identity(2*sites, dtype=np.complex128)

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
    if np.linalg.matrix_rank(M) < 2 * N:
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







