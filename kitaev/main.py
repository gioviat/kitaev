from kitaev.assemblers import kit_hamiltonian
from kitaev.assemblers import bath_operators
from kitaev.assemblers import dissipator


def main():
    mu = -0.4
    t = -2.0
    delta = -t
    gamma = 0.2
    sites = 64
    h = kit_hamiltonian(mu, t, delta, sites)
    l = bath_operators(gamma, sites)
    dissipator(h, l, sites)


if __name__ == '__main__':
    main()
