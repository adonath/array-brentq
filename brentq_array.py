# This is a vectorised version of Brent's method

from numpy import array_api as xp

def interpolate(fpre, fcur, xpre, xcur):
    """Interpolate."""
    return -fcur * (xcur - xpre) / (fcur - fpre)


def extrapolate(fpre, fcur, fblk, xpre, xcur, xblk):
    """Extrapolate."""
    dpre = (fpre - fcur) / (xpre - xcur)
    dblk = (fblk - fcur) / (xblk - xcur)
    return -fcur * (fblk * dblk - fpre * dpre) / (dblk * dpre * (fblk - fpre))


def brentq_array(f, xa, xb, args=(), xtol=2e-12, rtol=1e-15, maxiter=100):
    """Element wise Brent's method."""
    # adapted from https://github.com/scipy/scipy/blob/v1.11.3/scipy/optimize/Zeros/brentq.c
    # this is really ugly, but it works

    if xa.shape != xb.shape:
        raise ValueError("a and b must have the same shape")

    xpre, xcur = xa, xb

    fpre, fcur = f(xpre, *args), f(xcur, *args)

    n_iter = xp.zeros(xa.shape, dtype=xp.int32)
    xblk = xp.zeros(xa.shape, dtype=xp.float64)
    fblk = xp.zeros(xa.shape, dtype=xp.float64)
    spre = xp.zeros(xa.shape, dtype=xp.float64)
    scur = xp.zeros(xa.shape, dtype=xp.float64)

    if xa.shape != fpre.shape:
        raise ValueError("a and f(a) must have the same shape")

    # Check if the root is bracketed for all elements
    if xp.any(fpre * fcur >= 0):
        raise ValueError("f(a) and f(b) must have opposite signs in all elements")

    # If any of these is already zero, there is nothing to do...
    converged  = (fpre == 0) | (fcur == 0)

    for _ in range(maxiter):
        n_iter[~converged] += 1
        
        condition = (fpre != 0) & (fcur != 0) & (fpre * fcur < 0)
        xblk[condition] = xpre[condition]
        fblk[condition] = fpre[condition]
        spre[condition] = scur[condition] = xcur[condition] - xpre[condition]


        condition = xp.abs(fblk) < xp.abs(fcur) 
        xpre[condition] = xcur[condition]
        xcur[condition] = xblk[condition]
        xblk[condition] = xpre[condition]

        fpre[condition] = fcur[condition]
        fcur[condition] = fblk[condition]
        fblk[condition] = fpre[condition]

        delta = (xtol + rtol * xp.abs(xcur)) / 2.
        sbis = (xblk - xcur) / 2.

        condition = (fcur == 0) | (xp.abs(sbis) < delta)
        converged[condition] = True

        if xp.all(converged):
           break

        condition = (xp.abs(spre) > delta) & (xp.abs(fcur) < xp.abs(fpre))
        interp = condition & (xpre == xblk)
        stry = xp.where(interp, interpolate(fpre, fcur, xpre, xcur), extrapolate(fpre, fcur, fblk, xpre, xcur, xblk))

        bisect = condition & (2 * xp.abs(stry) < xp.min(xp.asarray([xp.abs(spre), 3 * xp.abs(sbis) - delta]), axis=0))
        spre = xp.where(bisect, scur, sbis)
        scur = xp.where(bisect, stry, sbis)

        spre[~condition] = sbis[~condition]
        scur[~condition] = stry[~condition]

        xpre, fpre = xcur, fcur

        condition = (xp.abs(scur) > delta) & ~converged
        xcur[condition] += scur[condition]
        xcur[~condition] += xp.where(sbis[~condition] > 0,  delta[~condition], -delta[~condition])

        fcur = f(xcur, *args)

    return xcur, n_iter, converged
