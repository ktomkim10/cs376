import numpy as np

def estimate_pi():
    sizes = [100, 500, 1000, 10000, 100000]
    pis = []
    np.random.seed(0)
    for size in sizes:
        counter = 0
        for i in range(size):
            xy = np.random.rand(1, 2)
            x = xy.item(0, 0)
            y = xy.item(0, 1)
            if x**2 + y**2 < 1:
                counter = counter + 1
        pi = 4 * float(counter) / size
        pis.append(pi)
    print pis
    return pis

if __name__ == "__main__":
    estimate_pi()
