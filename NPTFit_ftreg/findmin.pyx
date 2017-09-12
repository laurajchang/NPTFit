###############################################################################
# findmin.pyx
###############################################################################
#
# Return the minimum in an array 
#
###############################################################################

import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off bounds-checking for entire function
def minval(double[:] x, return_ind = False):
	cdef int index = 0
	cdef double minval = x[0]
	cdef Py_ssize_t j

	for j in range(1, len(x)):
		if x[j] < minval:
			minval = x[j]
			if return_ind:
				index = j

	if return_ind:
		return index, minval
	else: 
		return minval


