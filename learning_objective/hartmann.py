import numpy as np
from numpy import atleast_2d as vec

alpha = [1.0, 1.2, 3.0, 3.2]
A = vec([[10, 3, 17, 3.5, 1.7, 8],
     [0.05, 10, 17, 0.1, 8, 14],
     [3, 3.5, 1.7, 10, 17, 8],
     [17, 8, 0.05, 10, 0.1, 14]])
P = 10 ** (-4) * vec([[1312, 1696, 5569, 124, 8283, 5886],
                      [2329, 4135, 8307, 3736, 1004, 9991],
                      [2348, 1451, 3522, 2883, 3047, 6650],
                      [4047, 8828, 8732, 5743, 1091, 381]])


def hartmann(x):
    outer = 0.0
    for i in range(4):
        inner = 0.0
        for j in range(4):
            xj = x[0, j]
            Aij = A[i, j]
            Pij = P[i, j]
            inner += Aij * (xj - Pij) ** 2

        new = alpha[i] * np.exp(-inner)
        outer += new
    return vec(-1 * (1.1 - outer) / 0.839)


if __name__ == "__main__":
    print alpha
    print A
    print P

    print hartmann([0.5,0.5,0.5,0.5])
