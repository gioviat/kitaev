import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from assemblers import kit_hamiltonian, dissipator, correlation_matrix
from computers import compute_EGP, compute_particle_density

matplotlib.rcParams.update({'font.size': 12})


def plot_density_gamma(mu: float, t: float, delta: float, gamma_g1: float, gamma_g2: float,
                       gamma_points: int, gamma_l: float, sites: int):
    gamma_g_array = np.linspace(gamma_g1, gamma_g2, gamma_points)
    site_density = []
    for gamma_g in gamma_g_array:
        density = compute_particle_density(mu, t, delta, gamma_g, gamma_l, sites)
        site_density.append(density[0])
    plt.plot(gamma_g_array / gamma_l, site_density)
    plt.title('%d-site Kitaev chain with $\mu=%.2f$, $t=%.2f$, $\Delta=%.2f$' % (sites, mu, t, delta))
    plt.xlabel('$\gamma_g/\gamma_l$')
    plt.ylabel('Average density')
    plt.show()
    plt.close()


def plot_density(mu: float, t: float, delta: float, gamma_g: float, gamma_l: float, sites: int):
    density = compute_particle_density(mu, t, delta, gamma_g, gamma_l, sites)
    plt.scatter(np.arange(sites), density)
    plt.title('Particle density for each site in a Kitaev chain with\n'
              '$\mu=%.2f$, $t=%.2f$, $\Delta=%.2f$, $\gamma_g=%.2f$, $\gamma_l=%.2f$ '
              'and %d sites' % (mu, t, delta, gamma_g, gamma_l, sites))
    plt.ylim([-0.1, 1.2])
    plt.xticks(np.arange(0, sites, 1.0))
    plt.xlabel('Site')
    plt.ylabel('Average particle density')
    plt.show()
    plt.close()

    return


def plot_correlation(mu: float, t: float, delta: float, gamma_g: float, gamma_l: float,
                     sites: int):
    """
    Plots the real and imaginary parts of the correlation matrix for the Kitaev chain.
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
    H = kit_hamiltonian(mu, t, delta, sites)
    M = dissipator(gamma_g, gamma_l, sites)
    C = correlation_matrix(H, M, sites)
    plt.matshow(C.real, cmap='inferno')
    plt.colorbar()
    #plt.title('Real part of the correlation matrix\n'
    #         'for a Kitaev chain with $\mu=%.2f$, $t=%.2f$, $\Delta=%.2f$,\n'
    #        '$\gamma_g=%.2f$, $\gamma_l=%.2f$ and %d sites\n' % (mu, t, delta, gamma_g, gamma_l, sites))
    plt.matshow(C.imag, cmap='inferno')
    plt.colorbar()
    #plt.title('Imaginary part of the correlation matrix\n'
    #         'for a Kitaev chain with $\mu=%.2f$, $t=%.2f$, $\Delta=%.2f$,\n'
    #        '$\gamma_g=%.2f$, $\gamma_l=%.2f$ and %d sites\n' % (mu, t, delta, gamma_g, gamma_l, sites))
    plt.show()
    plt.close()

    return


def plot_egp(mu1: float, mu2: float, t: float, delta1: float, delta2: float, mu_points: int, delta_points: int,
             gamma_g: float, gamma_l: float, sites: int):
    """
Plots the EGP for a Kitaev chain when t, gamma_g and gamma_l are maintained fixed and mu and delta are varied.
    Args:
        mu1: lower bound of the mu parameter range
        mu2: upper bound of the mu parameter range
        t: hopping amplitude
        delta1: lower bound of the delta parameter range
        delta2: upper bound of the delta parameter range
        mu_points: number of values of mu to consider
        delta_points: number of values of delta to consider
        gamma_g: fermions injection rate
        gamma_l: fermions extraction rate
        sites: number of sites in the Kitaev chain

    Returns:
Nothing, but plots the EGP.
    """
    mu_array = np.linspace(mu1, mu2, mu_points)
    delta_array = np.linspace(delta1, delta2, delta_points)
    egp_array = np.zeros([mu_points, delta_points])

    for mu_idx, mu in enumerate(mu_array):
        for delta_idx, delta in enumerate(delta_array):
            u_real, u_imag = compute_EGP(mu=mu, t=t, delta=delta, gamma_g=gamma_g, gamma_l=gamma_l, sites=sites)
            egp_array[mu_idx, delta_idx] = np.imag(np.log(u_real + 1j * u_imag))
            print((mu_idx * delta_points + delta_idx) / (mu_points * delta_points) * 100, '% of calculation complete')

    x, y = np.meshgrid(delta_array / t, mu_array / t)
    z = egp_array / np.pi

    fig = plt.figure(figsize=(9, 6))
    fig.suptitle("EGP [$\pi$]")
    ax1 = fig.gca()
    contour = ax1.pcolormesh(x, y, np.arccos(np.cos(z)), cmap='viridis')
    ax1.grid()
    ax1.set_xlabel(r"$\Delta/t$")
    ax1.set_ylabel(r'$\mu/t$')
    cbar = [fig.colorbar(contour)]
    plt.show()
    plt.close()

    return


def plot_egp_mu(mu1: float, mu2: float, t: float, delta: float, mu_points: int,
                gamma_g1: float, gamma_g2: float, gamma_points: int, gamma_l: float, sites: int):
    """
Plots the EGP for a Kitaev chain when t, delta and gamma_l are maintained fixed and mu and gamma_g are varied.
    Args:
        mu1: lower bound of the mu parameter range
        mu2: upper bound of the mu parameter range
        t: hopping amplitude
        delta: superconducting gap
        mu_points: number of values of mu to consider
        gamma_g1: lower bound of the gamma_g parameter range
        gamma_g2: upper bound of the gamma_g parameter range
        gamma_points: number of values of gamma to consider
        gamma_l: fermions extraction rate
        sites: number of sites in the Kitaev chain

    Returns:
Nothing, but plots the EGP.
    """
    mu_array = np.linspace(mu1, mu2, mu_points)
    gamma_array = np.linspace(gamma_g1, gamma_g2, gamma_points)
    egp_array = np.zeros([mu_points, gamma_points])

    for mu_idx, mu in enumerate(mu_array):
        for gamma_idx, gamma_g in enumerate(gamma_array):
            u_real, u_imag = compute_EGP(mu=mu, t=t, delta=delta, gamma_g=gamma_g, gamma_l=gamma_l, sites=sites)
            egp_array[mu_idx, gamma_idx] = np.imag(np.log(u_real + 1j * u_imag))
            print((mu_idx * gamma_points + gamma_idx) / (mu_points * gamma_points) * 100, '% of calculation complete')

    x, y = np.meshgrid(mu_array / t, gamma_array / gamma_l)
    z = egp_array / np.pi

    fig = plt.figure(figsize=(9, 6))
    fig.suptitle("EGP [$\pi$]")
    ax1 = fig.gca()
    contour = ax1.pcolormesh(x, y, np.arccos(np.cos(z)), cmap='viridis')
    ax1.grid()
    ax1.set_xlabel(r"$\mu/t$")
    ax1.set_ylabel(r'$\gamma_g/\gamma_l$')
    cbar = [fig.colorbar(contour)]
    plt.show()
    plt.close()

    return


def plot_egp_t(mu: float, t1: float, t2: float, delta: float, t_points: int,
                gamma_g1: float, gamma_g2: float, gamma_points: int, gamma_l: float, sites: int):
    """
Plots the EGP for a Kitaev chain when mu, delta and gamma_l are maintained fixed and t and gamma_g are varied.
    Args:
        mu: onsite potential
        t1: lower bound of the t parameter range
        t2: upper bound of the t parameter range
        delta: superconducting gap
        t_points: number of values of t to consider
        gamma_g1: lower bound of the gamma_g parameter range
        gamma_g2: upper bound of the gamma_g parameter range
        gamma_points: number of values of gamma to consider
        gamma_l: fermions extraction rate
        sites: number of sites in the Kitaev chain

    Returns:
Nothing, but plots the EGP.
    """
    t_array = np.linspace(t1, t2, t_points)
    gamma_array = np.linspace(gamma_g1, gamma_g2, gamma_points)
    egp_array = np.zeros([t_points, gamma_points])

    for t_idx, t in enumerate(t_array):
        for gamma_idx, gamma_g in enumerate(gamma_array):
            u_real, u_imag = compute_EGP(mu=mu, t=t, delta=delta, gamma_g=gamma_g, gamma_l=gamma_l, sites=sites)
            egp_array[t_idx, gamma_idx] = np.imag(np.log(u_real + 1j * u_imag))
            print((t_idx * gamma_points + gamma_idx) / (t_points * gamma_points) * 100, '% of calculation complete')

    x, y = np.meshgrid(t_array / mu, gamma_array / gamma_l)
    z = egp_array / np.pi

    fig = plt.figure(figsize=(9, 6))
    fig.suptitle("EGP [$\pi$]")
    ax1 = fig.gca()
    contour = ax1.pcolormesh(x, y, np.arccos(np.cos(z)), cmap='viridis')
    ax1.grid()
    ax1.set_xlabel(r"$t/\mu$")
    ax1.set_ylabel(r'$\gamma_g/\gamma_l$')
    cbar = [fig.colorbar(contour)]
    plt.show()
    plt.close()

    return


def plot_egp_delta(mu: float, t: float, delta1: float, delta2: float, delta_points: int,
                gamma_g1: float, gamma_g2: float, gamma_points: int, gamma_l: float, sites: int):
    """
Plots the EGP for a Kitaev chain when mu, t and gamma_l are maintained fixed and delta and gamma_g are varied.
    Args:
        mu: onsite potential
        t: hopping amplitude
        delta1: lower bound of the delta parameter range
        delta2: upper bound of the delta parameter range
        delta_points: number of values of delta to consider
        gamma_g1: lower bound of the gamma_g parameter range
        gamma_g2: upper bound of the gamma_g parameter range
        gamma_points: number of values of gamma to consider
        gamma_l: fermions extraction rate
        sites: number of sites in the Kitaev chain

    Returns:
Nothing, but plots the EGP.
    """
    delta_array = np.linspace(delta1, delta2, delta_points)
    gamma_array = np.linspace(gamma_g1, gamma_g2, gamma_points)
    egp_array = np.zeros([delta_points, gamma_points])

    for delta_idx, delta in enumerate(delta_array):
        for gamma_idx, gamma_g in enumerate(gamma_array):
            u_real, u_imag = compute_EGP(mu=mu, t=t, delta=delta, gamma_g=gamma_g, gamma_l=gamma_l, sites=sites)
            egp_array[delta_idx, gamma_idx] = np.imag(np.log(u_real + 1j * u_imag))
            print((delta_idx * gamma_points + gamma_idx) / (delta_points * gamma_points) * 100, '% of calculation complete')

    x, y = np.meshgrid(delta_array / t, gamma_array / gamma_l)
    z = egp_array / np.pi

    fig = plt.figure(figsize=(9, 6))
    fig.suptitle("EGP [$\pi$]")
    ax1 = fig.gca()
    contour = ax1.pcolormesh(x, y, np.arccos(np.cos(z)), cmap='viridis')
    ax1.grid()
    ax1.set_xlabel(r"$\Delta/t$")
    ax1.set_ylabel(r'$\gamma_g/\gamma_l$')
    cbar = [fig.colorbar(contour)]
    plt.show()
    plt.close()

    return
