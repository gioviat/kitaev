from visualization import plot_correlation, plot_egp_gamma, plot_egp, plot_density
from computers import compute_EGP


def main():

    mu = 0.2
    t = 0.4
    tprime = 0.2
    tprime_1 = 0.01
    tprime_2 = 0.8
    tprime_points = 50
    delta1 = 0
    delta2 = 1
    delta_points = 100
    gamma_g1 = 0.1
    gamma_g2 = 0.7
    gamma_l = 0.5
    gamma_points = 50
    sites = 16

    plot_density(t, tprime, gamma_g1, gamma_l, sites)
    plot_correlation(t, tprime, gamma_g2, gamma_l, sites)
    plot_egp(t, tprime_1, tprime_2, tprime_points,
             gamma_g1=gamma_g1, gamma_g2=gamma_g2, gamma_l=gamma_l,
             gamma_points=gamma_points, sites=sites)


if __name__ == '__main__':
    main()
