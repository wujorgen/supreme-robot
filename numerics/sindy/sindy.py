import numpy as np


def sindy(X: np.array, X_dot: np.array, thresh=0.01, iterations=10) -> np.array:
    """Implementation of the SINDy algorithm using Python"""
    xi = np.linalg.lstsq(X, X_dot, rcond=None)[0]  # initial guess using least squares
    n = xi.shape[
        1
    ]  # n as in m x n of state vector -> in this example, its 1 x 3 so n = 3

    # begin sparsifying loop
    for k in range(iterations):
        small_indices = abs(xi) < thresh  # find coeffs less than threshold
        xi[small_indices] = 0  # set them to zero
        for ind in range(n):
            big_indices = ~small_indices[:, ind]
            xi[big_indices, ind] = np.linalg.lstsq(
                X[:, big_indices], X_dot[:, ind], rcond=None
            )[0]

    return xi


def disp_cf(xi, vars):
    cx, cy = xi.shape
    header_str = ""
    for var in vars:
        outstr = f"{var:^12}"
        header_str += outstr
        print(outstr, end=" | ")
    print()
    cfs = []
    for cdy in range(cy):
        cfstr = ""
        for cdx in range(cx):
            outstr = f"{xi[cdx,cdy]:>12.2f}"
            cfstr += outstr
            print(outstr, end=" | ")
        cfs.append(cfstr)
        print()
    return {"header": header_str, "cfs": cfs}
