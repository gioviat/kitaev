from kitaev.assemblers import kit_hamiltonian
from kitaev.assemblers import bath_operators
from kitaev.assemblers import dissipator


def main():
    mu = 5
    t = .7
    delta = .2
    loss = .63
    gain = .63
    sites = 20
    h = kit_hamiltonian(mu, t, delta, sites)
    l = bath_operators(loss, gain, sites)
    dissipator(h, l, sites)


if __name__ == '__main__':
    main()
