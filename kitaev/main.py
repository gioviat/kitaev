import numpy as np
import matplotlib.pyplot as plt
from visualization import plot_correlation, plot_egp_gamma
from computers import compute_gamma_transition


def main():

    mu = 0
    t = 0.6
    delta1 = 0.12
    delta2 = 0.24
    delta_points = 100
    gamma1 = 0.1
    gamma2 = 0.5
    gamma_points_correlation = 3
    gamma_points_egp = 100
    sites = 8

    # plot_correlation(mu, t, delta, gamma1, gamma2, gamma_points_correlation, sites)
    # plot_egp_gamma(mu, t, delta, gamma1, gamma2, gamma_points_egp, sites)

    delta_array = np.linspace(delta1, delta2, delta_points)
    gamma_array = np.zeros(delta_points)

    for i, delta in enumerate(delta_array):
        gamma_array[i] = compute_gamma_transition(mu, t, delta, gamma1, gamma2, gamma_points_egp, sites)

    plt.scatter(delta_array/t, gamma_array/t)
    plt.title('$\gamma$ of transition for various values of $\Delta/t$')
    plt.xlabel('$\Delta/t$')
    plt.ylabel('$\gamma_{trans}/t$')
    plt.show()

if __name__ == '__main__':
    main()
