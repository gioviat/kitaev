from visualization import (plot_egp, plot_density, plot_egp_mu, plot_egp_t, plot_egp_delta,
                           plot_density_gamma, plot_correlation)
from computers import compute_EGP
import numpy as np
from redfield import plot_EGP_phase_diagram, plot_Majorana_correlator, plot_filling_beta, plot_EGP_mu, plot_EGP_beta
import matplotlib.font_manager

import pylab as pl


def main():
    # mu1 = 0, mu2 = 2, t = 0.1 -> 2, delta = 1, gamma_g1 = 0, gamma_g2 = 2, gamma_l = 0.5
    # mu = 0.1 -> 2, t1 = 0, t2 = 2, delta = 1, gamma_g1 = 0, gamma_g2 = 2, gamma_l = 0.5
    # mu = 1, t = 0.1 -> 2, delta1 = 0, delta2 = 2, gamma_g1 = 0, gamma_g2 = 2, gamma_l = 0.5
    mu1 = 0
    mu2 = 5
    mu = 1
    t1 = 0
    t2 = 2
    t = 1
    delta1 = 0.01
    delta2 = 1.5
    delta = 0.1
    mu_points = 100
    t_points = 100
    delta_points = 100
    gamma_g = 1
    gamma_g1 = 0
    gamma_g2 = 2
    gamma_l = 0.2
    gamma_points = 100
    gamma_slices = 2
    sites = 16
    gammaA_l1 = 1
    gammaA_l2 = 2
    gammaA_g = 1
    gammaB_l = 0.2
    gammaB_g = 0.01
    tprime1 = 0
    tprime2 = 1
    tprime_points = 10

    #plot_density_gamma_temp(mu, t, delta, gamma_g1, gamma_g2, gamma_points, gamma_l, sites)
    #plot_entropy(mu, t, delta, gamma_g1, gamma_g2, gamma_l, gamma_points, sites)
    #plot_SSH_egp(t1, t2, tprime1, tprime2, t_points, tprime_points, gamma_g, gamma_l, sites)

    #plot_correlation(mu, t, delta, gammaA_l, gammaA_g, gammaB_l, gammaB_g, sites)
    #plot_density_gamma(mu, t, delta, gammaA_l1, gammaA_l2, gammaA_g, gammaB_l, gammaB_g, gamma_points, sites)

    #plot_egp_3d(mu1, mu2, t, delta1, delta2, mu_points, delta_points, gamma_g1, gamma_g2, gamma_l, gamma_slices, sites)
    #plot_gamma_transition(mu1, mu2, t, delta1, delta2, mu_points, delta_points, gamma_g1, gamma_g2, gamma_l,
                          #gamma_points, sites)
    plot_density(mu=mu, t=t, delta=delta, gamma_g=gamma_g, gamma_l=gamma_l, sites=sites)
    #plot_density_gamma(mu, t, delta, gamma_g1, gamma_g2, gamma_points, gamma_l, sites)
    #plot_correlation(mu, t, delta, gamma_g, gamma_l, sites)
    # gamma_array = np.linspace(gamma_g1, gamma_g2, 5)
    # for gamma_g in gamma_array:
    #     plot_egp(mu1=mu1, mu2=mu2, t=t, delta1=delta1, delta2=delta2,
    #              mu_points=mu_points, delta_points=delta_points,
    #              gamma_g=gamma_g, gamma_l=gamma_l, sites=sites)

    # gamma_g1 = 0.6
    # gamma_g2 = 2
    # gamma_array = np.linspace(gamma_g1, gamma_g2, 5)
    # for gamma_g in gamma_array:
    #     plot_egp(mu1=mu1, mu2=mu2, t=t, delta1=delta1, delta2=delta2,
    #              mu_points=mu_points, delta_points=delta_points,
    #              gamma_g=gamma_g, gamma_l=gamma_l, sites=sites)
    #plot_egp(mu1, mu2, t, delta1, delta2, mu_points, delta_points, gamma_g, gamma_l, sites)
    #plot_egp_mu(mu1, mu2, t, delta,mu_points, gamma_g1, gamma_g2, gamma_points, gamma_l, sites)
    #plot_egp_t(mu, t1, t2, delta, t_points, gamma_g1, gamma_g2, gamma_points, gamma_l, sites)
    #plot_egp_delta(mu, t, delta1, delta2, delta_points, gamma_g1, gamma_g2, gamma_points, gamma_l, sites)



    #### REDFIELD

    # 1.5, 1, 0.1
    # 0.5, 2, 1
    # 0.1, 2, 4
    mu = 1
    t = 0.1
    delta = 0.01
    lambd = 0.01
    betap_start = -5
    beta1 = betap_start
    betap_end = 5
    beta2 = beta1
    betap_points = 50
    mu_start = -3.0
    mu1 = 0
    mu_end = 3.0
    mu2 = 2
    mu_points = 50
    sites = 16
    load = False
    rounding_precision = 8
    redfield = False

    #plot_Majorana_correlator(mu, t, delta, lambd, beta1, mu1, beta2, mu2, sites, rounding_precision)
    #plot_filling_beta(mu, t, delta, lambd, beta1, mu_start, mu_end, mu_points, sites, rounding_precision)

    beta_array = [0.01, 1.0, 10.0, 100.0]
    #plot_EGP_mu(mu1, mu2, t, delta, mu_points, lambd, beta_array, chempot=0.0, sites=sites,
                #rounding_precision=rounding_precision)

    #chempot_array = [0.0]
    #plot_EGP_beta(mu, t, delta, lambd, betap_start, betap_end, betap_points, chempot_array, sites, rounding_precision)

    if redfield:
        plot_EGP_phase_diagram(mu=mu,
                               t=t,
                               delta=delta,
                               lambd=lambd,
                               betap_start=betap_start,
                               betap_end=betap_end,
                               betap_points=betap_points,
                               mu_start=mu_start,
                               mu_end=mu_end,
                               mu_points=mu_points,
                               sites=sites,
                               load=load,
                               rounding_precision=rounding_precision)


if __name__ == '__main__':
    main()
