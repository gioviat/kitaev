from visualization import plot_correlation, plot_egp_gamma


def main():

    mu = 0
    t = 0.6
    delta = 0.2
    gamma1 = 0.1
    gamma2 = 0.5
    gamma_points_correlation = 3
    gamma_points_egp = 100
    sites = 8

    plot_correlation(mu, t, delta, gamma1, gamma2, gamma_points_correlation, sites)
    plot_egp_gamma(mu, t, delta, gamma1, gamma2, gamma_points_egp, sites)


if __name__ == '__main__':
    main()
