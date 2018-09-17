import numpy as np
from math import sqrt

def mc_estimate_pi():
    sizes = [100, 500, 1000, 10000, 100000]
    pis = []
    np.random.seed(0)
    for size in sizes:
        sum = 0
        for i in range(size):
            x = np.random.rand(1, 1).item(0, 0)
            f_val = sqrt(1 - x**2)
            sum = sum + f_val
        pi = 4 * 1 / float(size) * sum
        pis.append(pi)
    print pis
    return pis

if __name__ == "__main__":
    mc_estimate_pi()
