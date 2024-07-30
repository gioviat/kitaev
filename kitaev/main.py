from visualization import (plot_egp, plot_density, plot_egp_mu, plot_egp_t, plot_egp_delta,
                           plot_density_gamma, plot_correlation)
from redfield import plot_EGP_phase_diagram, plot_Majorana_correlator, plot_filling_beta


def main():
    mu1 = 0
    mu2 = 1
    t = 0.1
    delta = 0.01
    mu_points = 100
    gamma_g1 = 0.01
    gamma_g2 = 4
    gamma_points = 100
    gamma_l = 1
    sites = 16

    plot_egp_mu(mu1, mu2, t, delta, mu_points, gamma_g1, gamma_g2, gamma_points, gamma_l, sites)

if __name__ == '__main__':
    main()
