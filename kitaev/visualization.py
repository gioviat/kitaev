import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from assemblers import SSH_hamiltonian, dissipator, correlation_matrix
from computers import compute_EGP, compute_particle_density

matplotlib.rcParams.update({'font.size': 8})


def plot_density(t: float, tprime: float, gamma_g: float, gamma_l: float, sites: int):
    density = compute_particle_density(t, tprime, gamma_g, gamma_l, sites)
    plt.scatter(np.arange(sites), density)
    #plt.title('Particle density for each site in a Kitaev chain with\n'
     #         '$\mu=%.2f$, $t=%.2f$, $\Delta=%.2f$, $\gamma_g=%.2f$, $\gamma_l=%.2f$ '
      #        'and %d sites' % (mu, t, delta, gamma_g, gamma_l, sites))
    plt.ylim([-0.1,1.2])
    plt.xlabel('Sites')
    plt.ylabel('Average particle density')
    plt.show()
    plt.close()

    return


def plot_correlation(t: float, tprime: float, gamma_g: float, gamma_l: float,
                     sites: int):
    """
    Plots the real and imaginary parts of the correlation matrix for the Kitaev chain..
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
    H = SSH_hamiltonian(t, tprime, sites, PBC=False)
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


def plot_egp(t: float, tprime1: float, tprime2: float, tprime_points: int, gamma_g1: float, gamma_g2: float,
             gamma_l: float, gamma_points: int, sites: int):

    tprime_array = np.linspace(tprime1, tprime2, tprime_points)
    gamma_array = np.linspace(gamma_g1, gamma_g2, gamma_points)
    egp_array = np.zeros([tprime_points, gamma_points])

    for tprime_idx, tprime in enumerate(tprime_array):
        for gamma_idx, gamma_g in enumerate(gamma_array):
            u_real, u_imag = compute_EGP(t=t, tprime=tprime, gamma_g=gamma_g, gamma_l=gamma_l, sites=sites)
            egp_array[tprime_idx, gamma_idx] = np.imag(np.log(u_real + 1j*u_imag))
            print((tprime_idx*gamma_points+gamma_idx)/(tprime_points*gamma_points)*100, '% of calculation complete')

    x, y = np.meshgrid(tprime_array/t, gamma_array/gamma_l)
    z = egp_array/np.pi
    fig = plt.figure(figsize=(9, 6))
    #fig.suptitle('EGP phase diagram for a Kitaev chain with %d sites and $\mu=%.2f$,\n'
     #            '$t=%.2f$, $\Delta_1=%.2f$, $\Delta_2=%.2f$, $\gamma_{g1}=%.2f$, $\gamma_{g2}=%.2f$ and '
      #           '$\gamma_l = %.2f$' % (sites, mu, t, delta1, delta2, gamma_g1, gamma_g2, gamma_l))
    ax1 = fig.gca()
    contour = ax1.pcolormesh(x, y, z, cmap='viridis')
    ax1.grid()
    ax1.set_xlabel(r"$t'/t$")
    ax1.set_ylabel(r'$\gamma_g/\gamma_l$')
    cbar = [fig.colorbar(contour)]
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
