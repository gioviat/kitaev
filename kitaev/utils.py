import numpy as np
import control as ct


def lyapunov(x, m):
    n = m.shape[0]
    q = x.shape[0]
    m = -m
    mvect = m.flatten('F').reshape(n*n, 1)
    s = np.kron(np.eye(q, q), x.T) + np.kron(x.T, np.eye(q, q))
    zvect = -np.matmul(np.linalg.inv(s), mvect)
    z = zvect.reshape(n, n, order='F')

    lhs = np.matmul(x.T, z) + np.matmul(z, x)
    print(lhs - q)

    return z
