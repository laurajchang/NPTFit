###############################################################################
# interp1d.pyx
###############################################################################
#
# 1d Linear interpolation, adapted from http://stackoverflow.com/a/13494757
#
###############################################################################


import numpy as np
cimport numpy as np
cimport cython

DTYPEf = np.float64
ctypedef np.float64_t DTYPEf_t

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off bounds-checking for entire function
cpdef interp1d(np.ndarray[DTYPEf_t, ndim=1] x, np.ndarray[DTYPEf_t, ndim=1] y,
               np.ndarray[DTYPEf_t, ndim=1] new_x):
    """
    interp1d(x, y, new_x)

    Performs linear interpolation in 1d.

    Parameters
    ----------
    x : 1-D ndarray (double type)
        Array containg the x (abcissa) values. Must be monotonically
        increasing.
    y : 1-D ndarray (double type)
        Array containing the y values to interpolate. Must have same length as x.
    x_new: 1-D ndarray (double type)
        Array with new abcissas to interpolate.

    Returns
    -------
    new_y : 1-D ndarray
        Interpolated values.
    """
    cdef int nx = x.shape[0]
    cdef int ny = y.shape[0]

    assert (nx == ny), \
        "x and y must have equal length!"
    cdef int i, j
    cdef np.ndarray[DTYPEf_t, ndim=1] new_y = np.zeros(nx, dtype=DTYPEf)

    for i in range(nx):
    # Loop through x_new
        if x[0] >= new_x[i]:
            new_y[i] = y[0]
            print("Certain values in new_x too small!")
        elif x[nx-1] <= new_x[i]:
            new_y[i] = y[nx-1]
            print("Certain values in new_x too large!")
        else:
            for j in range(1, nx):
            # Loop through x
                if x[j] > new_x[i]:
                 new_y[i] = (y[j] - y[j-1]) / (x[j] - x[j-1]) * (new_x[i] - x[j-1]) + y[j-1]
                 break
    return new_y