from visualization import plot_egp_gamma


def main():

    mu = 0.2
    t = 1.0
    delta = t
    gamma1 = 0.1
    gamma2 = 0.5
    gamma_points = 100
    sites = 32

    plot_egp_gamma(mu, t, delta, gamma1, gamma2, gamma_points, sites)


if __name__ == '__main__':
    main()
