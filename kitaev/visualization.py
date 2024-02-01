import numpy as np
import matplotlib.pyplot as plt
from computers import compute_EGP


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

    for i in range(gamma_points):
        ax1.scatter(gamma_array[i], z[i]/np.pi)
    ax1.set_title('Ensemble geometric phase for a model with $\mu=%.2f$,\n$t=%.2f$, $\Delta=%.2f$, '
                  '$\gamma_1=%.2f$, $\gamma_2=%.2f$ and $%d$ sites' % (mu, t, delta, gamma1, gamma2, sites))
    ax1.set_xlabel('$\gamma$')
    ax1.set_ylabel('EGP[$\pi$]')
    plt.show()
    plt.close()

    return
