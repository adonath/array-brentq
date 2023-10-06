from numpy import array_api as xp
from numpy.testing import assert_allclose
from scipy.optimize import brentq

from brentq_array import brentq_array


# TODO: add a few more test cases and functions...
def f(x, a):
    return x - a


def test_brentq_array():
    x_root = xp.asarray([1.2, 2.1, 3.6, 4.1, 5.04])
    a = x_root + 1.3
    b = x_root - 1.1

    x_ref = [
        brentq(f, float(a_), float(b_), args=(x_,)) for a_, b_, x_ in zip(a, b, x_root)
    ]
    x, n_iter, converged = brentq_array(f, a, b, args=(x_root,))

    assert_allclose(x, x_ref)
    assert_allclose(x, x_root)
    assert_allclose(n_iter, 2)
    assert_allclose(converged, True)


if __name__ == "__main__":
    test_brentq_array()
