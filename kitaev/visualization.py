import numpy as np
import matplotlib.pyplot as plt
from assemblers import kit_hamiltonian, bath_operators, dissipator, correlation_matrix
from computers import compute_EGP, compute_particle_density


def plot_density(mu: float, t: float, delta: float, gamma_g: float, gamma_l: float, sites: int):
    density = compute_particle_density(mu, t, delta, gamma_g, gamma_l, sites)
    plt.scatter(np.arange(sites), density)
    plt.title('Particle density for each site in a Kitaev chain with\n'
              '$\mu=%.2f$, $t=%.2f$, $\Delta=%.2f$, $\gamma_g=%.2f$, $\gamma_l=%.2f$ and %d sites' % (mu, t, delta, gamma_g, gamma_l, sites))
    #plt.ylim([-0.6,0.6])
    plt.show()
    plt.close()

    return


def plot_correlation(mu: float, t: float, delta: float, gamma1: float, gamma2: float, gamma_points: int, sites: int):
    """
    Plots the real and imaginary parts of the correlation matrix for the Kitaev chain, for various
    values of the coupling constant gamma, ranging from gamma1 to gamma2.
    Parameters
    ----------
    mu: onsite potential
    t: hopping amplitude
    delta: superconducting gap
    gamma1: lower bound of the gamma parameter range
    gamma2: upper bound of the gamma parameter range
    gamma_points: number of values of gamma to consider
    sites: number of sites in the Kitaev chain

    Returns
    -------
    Nothing, but plots the correlation matrix.
    """
    gamma_array = np.linspace(gamma1, gamma2, gamma_points)

    h = kit_hamiltonian(mu, t, delta, sites)

    fig, axs = plt.subplots(gamma_points, 2, figsize=(12, 12))
    fig.suptitle('Real and imaginary parts of the correlation matrix for the Kitaev chain\n'
                 'with $\mu=%.2f$, $t =%.2f$, $\Delta=%.2f$ and various choices of $\gamma$ ranging\n'
                 'from $\gamma_1=%.2f$ to $\gamma_2=%.2f$' % (mu, t, delta, gamma1, gamma2))
    for i in range(gamma_points):
        gamma = gamma_array[i]
        l = bath_operators(gamma, sites)
        z = dissipator(h, l)
        c = correlation_matrix(z, sites)
        c_real = c.real
        c_imag = c.imag
        ax1 = axs[i, 0].imshow(c_real)
        axs[i, 0].set_title('Real part, $\gamma=%.2f$' % gamma)
        fig.colorbar(ax1)
        ax2 = axs[i, 1].imshow(c_imag)
        axs[i, 1].set_title('Imaginary part, $\gamma=%.2f$' % gamma)
        fig.colorbar(ax2)
    plt.show()
    plt.close()

    return


def plot_egp(mu: float, t: float, delta1: float, delta2: float, delta_points: int, gamma1: float, gamma2: float, gamma_points: int, sites: int):

    delta_array = np.linspace(delta1, delta2, delta_points)
    gamma_array = np.linspace(gamma1, gamma2, gamma_points)
    egp_array = np.zeros([delta_points, gamma_points])

    for delta_idx, delta in enumerate(delta_array):
        for gamma_idx, gamma in enumerate(gamma_array):
            u_real, u_imag = compute_EGP(mu, t, delta, gamma, sites)
            egp_array[delta_idx, gamma_idx] = np.imag(np.log(u_real + 1j*u_imag))
            print((delta_idx*gamma_points+gamma_idx)/(delta_points*gamma_points)*100, '% of calculation complete')

    x, y = np.meshgrid(delta_array/t, gamma_array/t)
    z = egp_array/np.pi
    fig = plt.figure(figsize=(9, 6))
    fig.suptitle('EGP phase diagram for a Kitaev chain with %d sites and $\mu=%.2f$,\n'
                 '$t=%.2f$, $\Delta_1=%.2f$, $\Delta_2=%.2f$, $\gamma_1=%.2f$ and $\gamma_2=%.2f$' % (sites, mu, t, delta1, delta2, gamma1, gamma2))
    ax1 = fig.gca()
    contour = ax1.pcolormesh(x, y, z, cmap='viridis')
    ax1.grid()
    ax1.set_xlabel(r'$\Delta/t$')
    ax1.set_ylabel(r'$\gamma/t$')
    cbar = [fig.colorbar(contour, shrink=0.5, aspect=5)]
    plt.show()
    plt.close()

    return


def plot_egp_gamma(mu: float, t: float, delta: float, gamma1: float, gamma2: float, gamma_points: int, sites: int):
    """
    Plots the EGP for a Kitaev chain in the Lindblad approximations, with the coupling constants ranging
    from gamma1 to gamma2.
    Parameters
    ----------
    mu: onsite potential
    t: hopping amplitude
    delta: superconducting gap
    gamma1: lower bound of the gamma parameter range
    gamma2: upper bound of the gamma parameter range
    gamma_points: number of points in the gamma parameter range
    sites: number of sites in the Kitaev chain

    Returns
    -------
    Nothing, but plots the EGP.
    """
    gamma_array = np.linspace(gamma1, gamma2, gamma_points)
    egp_array = np.zeros([gamma_points])

    for gamma_idx, gamma in enumerate(gamma_array):
        egp_real, egp_imag = compute_EGP(mu, t, delta, gamma, sites)
        egp_array[gamma_idx] = np.imag(np.log(egp_real + 1j*egp_imag))

    fig = plt.figure(figsize=(9, 6))
    ax1 = fig.add_subplot(111)
    z = egp_array

    plt.axhline(y=0.5, linestyle='--')
    plt.axhline(y=-0.5, linestyle='--')
    for i in range(gamma_points):
        ax1.scatter(gamma_array[i], z[i]/np.pi, color='red')
    ax1.set_title('Ensemble geometric phase for a model with $\mu=%.2f$,\n$t=%.2f$, $\Delta=%.2f$, '
                  '$\gamma_1=%.2f$, $\gamma_2=%.2f$ and $%d$ sites' % (mu, t, delta, gamma1, gamma2, sites))
    ax1.set_xlabel('$\gamma$')
    ax1.set_ylabel('EGP[$\pi$]')
    plt.ylim([-0.6, 0.6])
    plt.show()
    plt.close()

    return
