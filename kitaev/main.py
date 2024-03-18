from visualization import plot_correlation, plot_egp_gamma, plot_egp, plot_density


def main():

    mu = 0.2
    t = 0.4
    delta1 = 0
    delta2 = 0
    delta_points = 2
    gamma_g1 = 0
    gamma_g2 = 2
    gamma_l = 0.5
    gamma_points_correlation = 1
    gamma_points_egp = 100
    sites = 16

    plot_density(mu, t, delta2, gamma_g2, gamma_l, sites)
    plot_correlation(mu, t, delta2, gamma_g2, gamma_l, delta_points, sites)
    plot_egp(mu, t, delta1, delta2, delta_points, gamma_g1, gamma_g2, gamma_l, gamma_points_egp, sites)


if __name__ == '__main__':
    main()
