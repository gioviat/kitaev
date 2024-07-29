import numpy as np
from assemblers import kit_hamiltonian
from utils import bath_spectral_func
import scipy.linalg as spla
from pfapack import pfaffian as pf
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import rc
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 12


def dissipator(mu: float, t: float, delta: float,
               lambd: float, beta1: float, mu1: float, beta2: float, mu2: float, sites: int) -> np.ndarray:
    """
    Construct the matrix encoding dissipation for a Kitaev chain
    connected to Markovian baths in the Redfield approximation.
    Args:
        mu: onsite potential
        t: hopping amplitude
        delta: superconducting gap
        lambd: system-bath coupling constant
        beta1: inverse temperature of the first bath
        mu1: chemical potential of the first bath
        beta2: inverse temperature of the second bath
        mu2: chemical potential of the first bath
        sites: number of sites

    Returns:

    """
    # Construct the Hamiltonian
    H = kit_hamiltonian(mu, t, delta, sites)
    # Obtain eigenvalues and eigenvectors
    eps, u = np.linalg.eig(H)
    # Keep only the eigenvectors with positive eigenvalues
    eps_sorted = np.sort(eps)
    eps_sorted = eps_sorted[sites:]
    u_sorted = u[:, eps.argsort()]
    u_sorted = u_sorted[:, sites:]
    # Store x_nu vectors in a list
    x_mu_list = []
    for i in range(2 * sites):
        xi = np.zeros(2 * sites, dtype=np.complex128)
        xi[i] = 1.0 / np.sqrt(2)
        x_mu_list.append(xi)
    # Initialize M
    M = np.zeros(shape=(2 * sites, 2 * sites), dtype=np.complex128)

    z_nu_list = []
    for nu in range(2 * sites):

        # Initialize a new vector everytime:
        z_nu = np.zeros(2 * sites, dtype=np.complex128)

        # Loop over all (positive) eigenvalues (see formula - we invert the order of the
        # mu and m sums to code the expression in an easier way):
        for m in range(sites):

            # Obtain first bath correlation matrix via subroutine:
            # The function somehow gives NaNs if we don't make the eigenvalues explicitly
            # real (which should be no problem given that H is hermitian)
            GammaPositive = bath_spectral_func(lambd, np.real(4 * eps_sorted[m]), beta1, mu1, beta2, mu2, sites)
            GammaNegative = bath_spectral_func(lambd, np.real(-4 * eps_sorted[m]), beta1, mu1, beta2, mu2, sites)

            # Since the bath correlation matrix is block diagonal, for
            # even index nu (starting from zero), the only nonzero elements in the row
            # are at position nu and nu+1; while for odd indices, the only nonzero
            # elements in the row are nu-1 and nu.
            if nu % 2 == 0:
                x_mu1 = x_mu_list[nu]
                GammaPos1 = GammaPositive[nu, nu]
                GammaNeg1 = GammaNegative[nu, nu]
                x_mu2 = x_mu_list[nu + 1]
                GammaPos2 = GammaPositive[nu, nu + 1]
                GammaNeg2 = GammaNegative[nu, nu + 1]
            elif nu % 2 == 1:
                x_mu1 = x_mu_list[nu - 1]
                GammaPos1 = GammaPositive[nu, nu - 1]
                GammaNeg1 = GammaNegative[nu, nu - 1]
                x_mu2 = x_mu_list[nu]
                GammaPos2 = GammaPositive[nu, nu]
                GammaNeg2 = GammaNegative[nu, nu]

            # Add the current summand (index m - there are only two nonzero entries for the mu sum so we just write that explicitly) to the total sum
            z_nu = z_nu + np.pi * (
                    GammaNeg1 * np.vdot(x_mu1, np.conj(u_sorted[:, m])) * u_sorted[:, m] +
                    GammaPos1 * np.vdot(x_mu1, u_sorted[:, m]) * np.conj(u_sorted[:, m]) +
                    GammaNeg2 * np.vdot(x_mu2, np.conj(u_sorted[:, m])) * u_sorted[:, m] +
                    GammaPos2 * np.vdot(x_mu2, u_sorted[:, m]) * np.conj(u_sorted[:, m])
            )

        # Save the final vector into the list of all z-vectors.
        z_nu_list.append(z_nu)

        # Add the kronecker product of x_{nu} . z_{nu} to the matrix M:
    for nu in range(2 * sites):
        M = M + np.kron(x_mu_list[nu], z_nu_list[nu]).reshape(2 * sites, 2 * sites)

        # Now the matrix M has been fully assembled and can be returned.
    return M


def redfieldian(mu: float, t: float, delta: float,
                lambd: float, beta1: float, mu1: float, beta2: float, mu2: float, sites: float) -> (np.ndarray, float):
    """
    Constructs the structure matrix A of the Redfield master equation.
    Args:
        mu: onsite potential
        t: hopping amplitude
        delta: superconducting gap
        lambd: system-bath coupling constant
        beta1: inverse temperature of the first bath
        mu1: chemical potential of the first bath
        beta2: inverse temperature of the second bath
        mu2: chemical potential of the second bath
        sites: number of sites of the Kitaev chain

    Returns:
    A: structure matrix
    A0: constant proportional to the identity
    """
    # First initialize A:
    A = np.zeros(shape=(4 * sites, 4 * sites), dtype=np.complex128)

    # Then get Hamiltonian and dissipation matrix:
    H = kit_hamiltonian(mu, t, delta, sites)
    M = dissipator(mu, t, delta, lambd, beta1, mu1, beta2, mu2, sites)

    # Now assemble A and A0 following the rules derived in third quantization:
    # !!! Note that matrix indices start from 0 in python !!!
    for j in range(2 * sites):
        for k in range(2 * sites):
            A[2 * j][2 * k] = -2 * 1j * H[j, k] - M[j, k] + M[k, j]
            A[2 * j][2 * k + 1] = 1j * M[k, j] + 1j * np.conj(M[j, k])
            A[2 * j + 1][2 * k] = -1j * M[j, k] - 1j * np.conj(M[k, j])
            A[2 * j + 1][2 * k + 1] = -2 * 1j * H[j, k] - np.conj(M[j, k]) + np.conj(M[k, j])
    A0 = np.trace(M) + np.trace(np.conj(M))

    return A, A0


def compute_master_normal_modes(mu: float, t: float, delta: float, lambd: float,
                                beta1: float, mu1: float, beta2: float, mu2: float, sites: int,
                                rounding_precision=8, verbose=False) -> np.ndarray:
    """
    Diagonalize the Redfieldian structure matrix A to find the normal master modes.
    Args:
        mu: onsite potential
        t: hopping amplitude
        delta: superconducting gap
        lambd: system-bath coupling constant
        beta1: inverse temperature of the first bath
        mu1: chemical potential of the first bath
        beta2: inverse temperature of the second bath
        mu2: chemical potential of the second bath
        sites: number of sites in the Kitaev chain
        rounding_precision: number of floating point numbers to keep when rounding the rapidities
        verbose: prints out more information on the screen

    Returns:
    V, the matrix containing the coefficients of the normal master modes.
    """
    # Initialize the degeneracy flag:
    degeneracy_flag = False

    # Initialize matrix V:
    V = np.zeros(shape=(4 * sites, 4 * sites), dtype=np.complex128)

    # Get the structure matrix:
    A, A0 = redfieldian(mu, t, delta, lambd, beta1, mu1, beta2, mu2, sites)

    if verbose == True:
        # Print out the structure matrix:
        Aprint = np.round(A, 6)
        print('\n'.join([''.join(['{:16}'.format(item) for item in row]) for row in Aprint]))
        print('\n')

    # Print out some warning if the structure matrix is not numerically antisymmetric:
    if np.linalg.norm(A + A.T) > 10 ** (-10):
        print("Structure matrix A not perfectly antisymmetric!")
        return

    # Determine the eigenvalues of the structure matrix A.
    # EVs[:,i] is eigenvector to eigenvalue EWs[i]
    EWs, EVs = np.linalg.eig(A)
    # Round the eigenvalues to perform some sorting later:
    EWs = np.round(EWs, rounding_precision)
    # Sort eigenvalues from smallest to largest (sorting associated EW correspondigly)
    EWs_sorted = np.sort(EWs)
    EVs_sorted = EVs[:, EWs.argsort()]
    # Trick to fix some issues when the eigenvalues are purely real due to the edge states:
    if np.imag(EWs_sorted[-1]) < 10 ** (-10):
        temp = EWs_sorted[-2]
        EWs_sorted[-2] = EWs_sorted[-1]
        EWs_sorted[-1] = temp
        temp2 = EVs_sorted[:, -2]
        EVs_sorted[:, -2] = EVs_sorted[:, -1]
        EVs_sorted[:, -1] = temp2
    if verbose == True:
        # Prints the sorted eigenvalues and the eigenvectors on screen:
        print('\n'.join(['{:16}'.format(item) for item in EWs_sorted]))
        print('\n')
        for i in range(4 * sites):
            print(EVs_sorted[:, i])

    # If there are degeneracies we need to perform an additional orthonormalization.
    # To determine if this is the case, we first find the unique eigenvalues
    # and sort them from smallest to largest (first real part, then imaginary part, eg:
    # [-1 - 1j, -1, -1 + 1j, 1 - 1j, 1, 1 + 1j])
    EWs_unique = np.sort(np.unique(EWs_sorted))
    if verbose == True:
        print(f"Unique eigenvalues: {EWs_unique}")
    if len(EWs_unique) < len(EWs):
        degeneracy_flag = True

    # If there is no degeneracy, we can construct the matrix V directly:
    if degeneracy_flag == False:
        for i in range(2 * sites):
            V[2 * i, :] = np.copy(EVs_sorted[:, i])
            V[2 * i + 1, :] = np.copy(EVs_sorted[:, 4 * sites - 1 - i])
        # Orthonormalization for non-degenerate case:
        for i in range(2 * sites):
            V[2 * i, :] = np.copy(V[2 * i] / (np.dot(V[2 * i, :], V[2 * i + 1])))

    # If the eigenvalues (rapidities) are degenerate, we need to orthonormalize
    # the subspace of each eigenvalue. Typically, the SSH model has only double
    # degeneracies.
    if degeneracy_flag == True:
        # Creating a list where to store all the occurrences of different EWs:
        list_of_indices_of_same_EWs = []
        for l in range(len(EWs_unique)):
            list_of_indices_of_same_EWs.append([])

        # Looping through all EWs and sorting their *index* according to their value
        # (group the indices of the same eigenvalues):
        thresh = 10 ** (-6)
        for i_idx, i in enumerate(EWs_sorted):
            for j_idx, j in enumerate(EWs_unique):
                if np.abs(np.real(i) - np.real(j)) < thresh and np.abs(np.imag(i) - np.imag(j)) < thresh:
                    list_of_indices_of_same_EWs[j_idx].append(i_idx)

        if verbose == True:
            print("list of lists", list_of_indices_of_same_EWs)

        # Now that we know which eigenvectors are degenerate, we can start building the correct
        # superpositions:

        # The original EWs were sorted as [-b1, ..., -b1, -b2, ..., -b2, ..., b2, ... b2, b1, ..., b1]
        # with their corresponding multiplicities.
        # The eigenvectors are thus listed as (convention)
        # [v_{1,1}^-, v_{1,2}^-, ... v_{1,k1}^-, v_{2,1}^-, ... , v_{2,k2}^-, ... v_{2,1}^+, ..., v_{2,1}^+, v_{1,k1}^+, ..., v_{1,1}^+]
        # where v_j^+ is the j-th eigenvector of bj and v_j^- is the j-th eigenvector of -bj, and
        # k1 is the multiplicity of b1 (same as for -b1), k2 is the multiplicity of b2 (same as for -b2) etc.

        # Because of the degeneracies, we have to find new linear combinations for the v_j^-.
        # The transformation matrix for each degenerate eigenvector is given by C_{ij} = v_j^+ . v_i^-
        # (with bilinear scalar product and not sesquilinear! -> use np.dot and not np.vdot)

        EVs_sorted_normalized = np.copy(EVs_sorted)
        degeneracy_counter = 0
        for EW_idx_list in list_of_indices_of_same_EWs[:int(len(list_of_indices_of_same_EWs) / 2)]:
            # Initialize current transformation matrix C
            C = np.zeros(shape=(len(EW_idx_list), len(EW_idx_list)), dtype=np.complex128)

            for i_idx, i in enumerate(EW_idx_list):
                for j_idx, j in enumerate(EW_idx_list):
                    C[i_idx, j_idx] = np.dot(EVs_sorted[:, (4 * sites - 1) - degeneracy_counter - j_idx],
                                             EVs_sorted[:, degeneracy_counter + i_idx])
                    if verbose == True:
                        print(i_idx, j_idx, 4 * sites - 1 - degeneracy_counter - j_idx, degeneracy_counter + i_idx)

            # Invert the matrix C and obtain the new eigenvectors as a linear superposition of the old ones:
            # vnew_i^- = sum_j Cinv_{ij}*v_j^-
            Cinv = np.linalg.inv(C)
            # print("Current inverse C matrix: \n ", Cinv)
            for i_idx, i in enumerate(EW_idx_list):
                # reset the v_j^- eigenvalues:
                EVs_sorted_normalized[:, degeneracy_counter + i_idx] = 0.0
                for j_idx, j in enumerate(EW_idx_list):
                    EVs_sorted_normalized[:, degeneracy_counter + i_idx] += Cinv[i_idx, j_idx] * EVs_sorted[:,
                                                                                                 degeneracy_counter + j_idx]
                    if verbose == True:
                        print(degeneracy_counter + i_idx, i_idx, j_idx, degeneracy_counter + j_idx)

            degeneracy_counter += len(EW_idx_list)

        # Now assemble the matrix V from the correctly normalized eigenvectors:
        for i in range(2 * sites):
            V[2 * i, :] = np.copy(EVs_sorted_normalized[:, 4 * sites - 1 - i])  # v_j^+
            V[2 * i + 1, :] = np.copy(EVs_sorted_normalized[:, i])  # v_j^-

    # Check that the matrix V has been constructed correctly by comparing V.V^T against J:
    J = np.dot(V, V.T)
    Jref = np.kron(np.eye(2 * sites), np.array([[0, 1], [1, 0]]))

    if np.linalg.norm(J - Jref) > 0.1:
        print('Issues with normalization of V!')

    if verbose == True:
        print("Elements affected:")
        for i in range(np.shape(J)[0]):
            for j in range(np.shape(J)[1]):
                if np.abs(J[i, j] - Jref[i, j]) > 10 ** (-8):
                    print(i, j, J[i, j], Jref[i, j])

    # Print the matrices V and J
    if verbose == True:
        J = np.round(J, 3)
        print('\n'.join([''.join(['{:16}'.format(item) for item in row]) for row in J]))
        print('\n')

        V = np.round(V, 3)
        print('\n'.join([''.join(['{:16}'.format(item) for item in row]) for row in V]))
        print('\n')
        print('\n'.join([''.join(['{:16}'.format(item) for item in row]) for row in V.T]))

    return V


def compute_Majorana_correlator_element(V: np.ndarray, sites: int, j: int, k: int) -> float:
    """
    Calculates the correlator <w_j w_k> in the NESS with the matrix V in the formalism of third quantization.
    Args:
        V: matrix of normal master modes
        sites: number of sites in the Kitaev chain
        j: site j
        k: site k
    Returns:
    corr, the Majorana correlator <w_j w_k>.
    """
    # Formula for the Majorana correlator:
    corr = 0
    for m in range(2 * sites):
        corr += 2 * V[2 * m + 1, 2 * j] * V[2 * m, 2 * k]

    return corr


######################################################################################


######################################################################################
def compute_Majorana_correlator_full(V: np.ndarray, sites: int, rounding_precision: int) -> np.ndarray:
    """
    Calculates the full correlation matrix M, with elements M_jk = <w_j w_k>, from the matrix V
    of third quantization.
    Args:
        V: matrix of normal master modes
        sites: number of sites in the Kitaev chain
        rounding_precision: number of floating point numbers to keep when rounding the rapidities

    Returns:
    M, the correlation matrix in Majorana representation.
    """
    # Building the correlation matrix M, with M_ij = i*(<w_i w_j>_NESS - delta_{ij}):
    # the matrix M is antisymmetric, so we need to only compute the elements with i<j:
    M = np.zeros(shape=(2 * sites, 2 * sites), dtype=np.complex128)
    for i in range(2 * sites):
        for j in range(i + 1, 2 * sites):
            M[i][j] = 1j * compute_Majorana_correlator_element(V, sites, i, j,
                                                               rounding_precision)  # use subroutine to assemble matrix M with third quantisation
    # Get the other lower diagonal elements by taking the transpose:
    M = M - M.T

    return M


def compute_particle_density(mu: float, t: float, delta: float,
                             lambd: float, beta1: float, mu1: float, beta2: float, mu2: float,
                             sites: int, rounding_precision: int) -> np.ndarray:
    """
    Calculates the average particle density N_i on each site of the Kitaev chain.
    Args:
        mu: onsite potential
        t: hopping amplitude
        delta: superconducting gap
        lambd: system-bath coupling constant
        beta1: inverse temperature of the first bath
        mu1: chemical potential of the first bath
        beta2: inverse temperature of the second bath
        mu2: chemical potential of the second bath
        sites: number of sites of the Kitaev chain
        rounding_precision: number of floating point numbers to keep when rounding the rapidities

    Returns:
    dens, the array containing the average density for each site.
    """
    N = sites
    # First obtain matrix V from third quantisation:
    V = compute_master_normal_modes(mu, t, delta, lambd, beta1, mu1, beta2, mu2, sites, rounding_precision)

    # Initialize density array:
    dens = np.zeros(N, dtype=np.complex128)

    for i in range(N):
        # Formula for the particle density at site i:
        sum = 0
        for m in range(2 * sites):
            sum += V[2 * m + 1, 4 * i] * V[2 * m, 4 * i + 2]
        dens[i] = 0.5 + 1j * sum

    return dens


def compute_EGP(mu, t, delta, lambd, beta1, mu1, beta2, mu2, sites, rounding_precision):
    N = sites

    # Quantities needed:
    #   - correlation matrix M for the NESS.
    #   - matrix representation Ktilde and K2tilde of the operator U used in the definition of the EGP.

    # First obtain matrix V from third quantization to compute the Majorana correlator later:
    V = compute_master_normal_modes(mu, t, delta, lambd, beta1, mu1, beta2, mu2, sites, rounding_precision)

    # Build the correlation matrix M, with M_ij = i*(<w_i w_j>_NESS - delta_{ij}):
    M = compute_Majorana_correlator_full(V, N, rounding_precision)

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


def plot_EGP_phase_diagram(mu, t, delta, lambd, betap_start, betap_end, betap_points, mu_start, mu_end, mu_points, sites,
                           load=False, rounding_precision=8):
    # Initialize arrays:
    beta_array = np.exp(np.linspace(betap_start, betap_end, betap_points))
    mu_array = np.linspace(mu_start, mu_end, mu_points)
    EGP_array = np.zeros(shape=(mu_points, betap_points))


    # Obtain the EGP at every point in parameter space:
    if load == False:
        for beta_idx, beta in enumerate(beta_array):
            for mu_idx, mu in enumerate(mu_array):
                U_real, U_imag = compute_EGP(mu, t, delta, lambd, beta, mu, beta, mu, sites, rounding_precision)
                EGP_array[mu_idx, beta_idx] = np.imag(np.log(U_real + 1j * U_imag))
                print("Working on point (beta, mu)=(", beta, ",", mu, ").",
                      (beta_idx * mu_points + mu_idx) / (betap_points * mu_points) * 100,
                      " % of calculation completed.")


    # Plotting part:
    # Construct mesh grid:
    X, Y = np.meshgrid(beta_array, mu_array)
    Z = EGP_array

    # Create figure
    fig = plt.figure()
    ax1 = fig.gca()

    # Plot the heatmap:
    contour = ax1.pcolormesh(X, Y, np.arccos(np.cos(Z)), cmap='viridis', vmin=-np.pi, vmax=np.pi)
    ax1.set_xlabel(r'$\beta_1=\beta_2$')
    ax1.set_ylabel(r'$\mu_1=\mu_2$')
    ax1.set_xscale('log')
    fig.suptitle("EGP [$\pi$]")
    ax1.grid()
    cbar = [fig.colorbar(contour, shrink=0.5, orientation='horizontal', pad=0.2)]

    # Show graph:
    plt.show()
    plt.close()

    return


def plot_Majorana_correlator(mu, t, delta, lambd, beta1, mu1, beta2, mu2, sites, rounding_precision):
    N = sites

    # Obtain matrix V from third quantization and compute correlator:
    V = compute_master_normal_modes(mu, t, delta, lambd, beta1, mu1, beta2, mu2, N, rounding_precision)
    M = compute_Majorana_correlator_full(V, sites, rounding_precision)

    # Construct mesh grid:
    plt.matshow(M.imag, cmap='inferno', vmin=-0.1, vmax=.1)
    plt.colorbar()
    plt.show()
    plt.close()

    return


def plot_filling_beta(mu, t, delta, lambd, beta, mu_start, mu_end, mu_points, sites, rounding_precision):
    # Initialize arrays and lists:
    mu_array = np.linspace(mu_start, mu_end, mu_points)
    density_list = []

    # Compute the average density for every value of mu2:
    for mu_idx, mu in enumerate(mu_array):
        density_array = compute_particle_density(mu, t, delta, lambd, beta, mu, beta, mu, sites,
                                                     rounding_precision)
        density_list.append(np.sum(density_array) / sites)
        print(f"{(mu_idx + 1) / mu_points * 100} of program complete.")

    # Create figure
    fig = plt.figure(figsize=(9, 6))
    ax1 = fig.gca()
    fig.suptitle(r'System filling')

    # Plot the curves
    plt.plot(mu_array, density_list, color='blue', label='Filling')
    # Set labels:
    ax1.set_xlabel(r'$\mu_1=\mu_2$')
    ax1.set_ylabel(r'$\left<\hat{N}\right>$')
    ax1.set_ylim([-0.1, 1.1])
    ax1.grid(which='both')
    plt.tight_layout()
    plt.show()
    plt.close()

    return


def plot_EGP_mu(mu1, mu2, t, delta, mu_points, lambd, beta_array, chempot, sites,
                    rounding_precision=8):

    # Initialize arrays:
    mu_array = np.linspace(mu1, mu2, mu_points)
    muratio_array = mu_array / t
    cases = len(beta_array)
    EGP_array = np.zeros(shape=(mu_points, cases))

    # Loop over all points to generate the data:
    for i in range(cases):
        for mu_idx, mu in enumerate(mu_array):
            EGP_real, EGP_imag = compute_EGP(mu, t, delta, lambd, beta_array[i], chempot, beta_array[i], chempot, sites,
                                             rounding_precision)
            EGP_array[mu_idx, i] = np.angle(EGP_real + 1j * EGP_imag)
            print((mu_idx + mu_points * i) / (mu_points * cases) * 100, " % of calculation completed.")

    # Create figure
    fig = plt.figure(figsize=(9, 6))
    ax1 = fig.add_subplot(111)
    Z = EGP_array

    # Plot the curves:
    for i in range(cases):
        plot1 = ax1.scatter(muratio_array, Z[:, i] / np.pi, label=r'$\beta_1=\beta_2=$%s' % (beta_array[i]))
    ax1.set_ylabel(r"$\mathrm{EGP} [\pi]$")
    #ax1.set_ylim([-1.1, 1.1])
    ax1.grid()
    ax1.set_xlabel(r"$\mu/t$")
    plt.legend(loc='best')
    plt.show()
    plt.close()


def plot_EGP_beta(mu, t, delta, lambd, betap_start, betap_end, betap_points, chempot_array, sites,
                  rounding_precision=8):

    # Initialize arrays:
    beta_array = np.exp(np.linspace(betap_start, betap_end, betap_points))
    cases = len(chempot_array)
    EGP_array = np.zeros(shape=(betap_points, cases))

    for i in range(cases):
        for beta_idx, beta in enumerate(beta_array):
            EGP_real, EGP_imag = compute_EGP(mu, t, delta, lambd, beta, chempot_array[i], beta, chempot_array[i], sites,
                                             rounding_precision)
            EGP_array[beta_idx, i] = np.imag(np.log(EGP_real + 1j * EGP_imag))

            print("Working on point beta1=beta2=", beta, ",  mu1=mu2=", chempot_array[i], ".",
                    (beta_idx + betap_points * i) / (betap_points * cases) * 100, " % of calculation completed.")

    # create figure
    fig = plt.figure(figsize=(9, 6))
    ax1 = fig.add_subplot(111)
    Z = EGP_array

    # Plot the curves
    for i in range(cases):
        plot = ax1.scatter(beta_array, np.arccos(np.cos(Z[:, i])) / np.pi,
                           label=r'$\mu_1=\mu_2=$%s' % (chempot_array[i]))
    plt.hlines(1.0, beta_array[0], beta_array[-1], colors='red')
    plt.hlines(-1.0, beta_array[0], beta_array[-1], colors='red')

    # Set labels etc.
    ax1.set_ylabel(r"EGP [$\pi$]")
    ax1.set_ylim([-1.1, 1.1])
    ax1.grid(which='both')

    ax1.set_xlabel(r"$\beta_A=\beta_B$")
    ax1.set_xscale('log')
    plt.legend(loc='best')
    plt.show()
    plt.close()

    return






