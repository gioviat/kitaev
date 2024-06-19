from visualization import plot_correlation, plot_egp, plot_density, plot_egp_mu, plot_egp_t, plot_egp_delta
from computers import compute_EGP


def main():
    # mu = 0.2, t = 0.4, delta1 = 0, delta2 = 0.5
    # mu1 = 0, mu2 = 1, t = 0.2, delta = 0.1
    #
    mu1 = 0
    mu2 = 2
    mu = 1
    t1 = 0
    t2 = 2
    t = 1
    delta1 = 0
    delta2 = 2
    delta = 1
    mu_points = 100
    t_points = 100
    delta_points = 100
    gamma_g = 1
    gamma_g1 = 0
    gamma_g2 = 2
    gamma_l = 0.5
    gamma_points = 100
    sites = 16

    # plot_density(mu=mu, t=t, delta=delta, gamma_g=gamma_g, gamma_l=gamma_l, sites=sites)
    # plot_correlation(t, tprime, gamma_g2, gamma_l, sites)
    plot_egp(mu1=mu1, mu2=mu2, t=t, delta1=delta1, delta2=delta2,
             mu_points=mu_points, delta_points=delta_points,
             gamma_g=gamma_g, gamma_l=gamma_l, sites=sites)
    # plot_egp_mu(mu1, mu2, t, delta,mu_points, gamma_g1, gamma_g2, gamma_points, gamma_l, sites)
    # plot_egp_t(mu, t1, t2, delta, t_points, gamma_g1, gamma_g2, gamma_points, gamma_l, sites)
    # plot_egp_delta(mu, t, delta1, delta2, delta_points, gamma_g1, gamma_g2, gamma_points, gamma_l, sites)


if __name__ == '__main__':
    main()
