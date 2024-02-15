from visualization import plot_correlation, plot_egp_gamma, plot_egp, plot_density


def main():

    mu = 0
    t = 0.5
    delta1 = 0
    delta2 = 0.6
    delta_points = 50
    gamma1 = 0.1
    gamma2 = 0.6
    gamma_points_correlation = 3
    gamma_points_egp = 50
    sites = 8

    #plot_density(mu, t, delta2, gamma2, sites)
    #plot_correlation(mu, t, delta2, gamma1, gamma2, gamma_points_correlation, sites)
    plot_egp(mu, t, delta1, delta2, delta_points, gamma1, gamma2, gamma_points_egp, sites)


if __name__ == '__main__':
    main()
