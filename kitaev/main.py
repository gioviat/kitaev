from computers import compute_EGP


def main():

    mu = -0.2
    t = -1.0
    delta = -t
    gamma = 0.2
    sites = 32

    compute_EGP(mu, t, delta, gamma, sites)


if __name__ == '__main__':
    main()
