import time

import matplotlib.pyplot as plt
import numpy as np
from numpy import array_api as xp
from scipy.optimize import brentq

from brentq_array import brentq_array

n_pix = 2 ** np.arange(23)


def f(x, a):
    return x - a


def brentq_scipy(f, a, b, **kwargs):
    result = np.zeros_like(a)
    it = np.nditer(result, flags=["multi_index"])

    while not it.finished:
        value = kwargs["args"][0][it.multi_index]
        res = brentq(f, a[it.multi_index], b[it.multi_index], args=(value,))
        result[it.multi_index] = res
        it.iternext()

    return result


if __name__ == "__main__":
    times = {"brentq_scipy": [], "brentq_array": []}

    for n in n_pix:
        x_root = xp.asarray(np.random.uniform(-1, 1, size=(n,)))
        a = x_root + 1.3
        b = x_root - 1.1

        for func in [brentq_scipy, brentq_array]:
            t0 = time.time()
            func(f, a, b, args=(x_root,), maxiter=10)
            t1 = time.time()
            times[func.__name__].append(t1 - t0)
            print(f"Runtime for {func.__name__} with n_pix={n}: {t1 - t0:.3f} s")

    plt.plot(n_pix, times["brentq_scipy"], label="brentq_scipy")
    plt.plot(n_pix, times["brentq_array"], label="brentq_array")
    plt.loglog()
    plt.xlabel("n_pix")
    plt.ylabel("time [s]")
    plt.legend()
    plt.savefig("benchmark.png")
