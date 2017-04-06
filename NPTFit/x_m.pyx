###############################################################################
# x_m.pyx
###############################################################################
#
# Calculates x_m and x_m_sum arrays for arbitary number of breaks.
# Uses dedicated functions for 1, 2 and 3 breaks as well as n > 3 breaks.
#
###############################################################################

import numpy as np
cimport numpy as np
cimport cython
from cpython cimport bool

# from libc.math cimport fabs

try:
    import incgamma_fct as igf
    print("Using C x_m")
except ImportError:
    import incgamma_fct_p as igf
    print("Using Python x_m")

import interp1d as interp
import findmin

# Type used for all non-integer functions
DTYPE = np.float

# Setup cython functions used
cdef extern from "math.h":
    double pow(double x, double y) nogil


def return_xs(double[::1] theta, double[::1] f_ary, double[::1] df_rho_div_f_ary, 
              double[::1] ft_compressed, double[::1] npt_compressed, int[::1] data):
    """ Returns arrays of x_m and x_m_sum for likelihood calculations

        :param theta: Array of parameters, [A, n[1], .., n[j+1], Sb[1], .., Sb[j]]
        :param f_ary: Photon leakage probabilities characterizing PSF, sum(f_ary) = 1.0
        :param df_rho_div_f_ary: df*rho(f)/f for integrating over f as a sum
        :param ft_compressed: Pixel-wise flux template. For a nontrivial flux template, scaling
        flux by a factor of c corresponds to scaling normalization A by 1/c and breaks Sb[j] by c.
        :param npt_compressed: Pixel-wise normalization of the PS template
        :param data: The pixel-wise data
        :returns: A list containing (x_m, x_m_sum)
    """

    cdef int n_break = int((len(theta) - 2)/2) 

    if n_break == 1:
        return return_xs_1break(theta, f_ary, df_rho_div_f_ary, ft_compressed, npt_compressed,
                                data)
    elif n_break == 2:
        # return return_xs_2break_orig(theta, f_ary, df_rho_div_f_ary, npt_compressed, data)
        return return_xs_2break(theta, f_ary, df_rho_div_f_ary, ft_compressed, npt_compressed,
                                data)
    elif n_break == 3:
        return return_xs_3break(theta, f_ary, df_rho_div_f_ary, ft_compressed, npt_compressed,
                                data)
    else:
        return return_xs_lbreak(theta, n_break, f_ary, df_rho_div_f_ary, 
                                npt_compressed, data)
        print("Sorry, flux template not yet implemented!")

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
def return_xs_1break(double[::1] theta, double[::1] f_ary, double[::1] df_rho_div_f_ary, 
                     double[::1] ft_compressed, double[::1] npt_compressed, int[::1] data):
    """ Dedicated calculation of x_m and x_m_sum for 1 break 
    """

    cdef double a_ps = float(theta[0])
    cdef double n1 = float(theta[1])
    cdef double n2 = float(theta[2])
    cdef double sb = float(theta[3])

    cdef int k_max = int(max(data) + 1)
    cdef int npix_roi = len(npt_compressed)

    cdef double[:,::1] x_m_ary = np.zeros((npix_roi,k_max + 1), dtype=DTYPE)
    cdef double[::1] x_m_sum = np.zeros(npix_roi, dtype=DTYPE)

    cdef double[::1] g1_ary_f = np.empty(k_max + 1, dtype=DTYPE)
    cdef double[::1] g2_ary_f = np.empty(k_max + 1, dtype=DTYPE)

    cdef double f2, df_rho_div_f2
    cdef double pref1_x_m_ary, pref2_x_m_ary
    cdef double x_m_sum_f, x_m_ary_f
    cdef double minval1, minval2

    cdef Py_ssize_t f_index, p, k, i

    cdef double[:] sb_ref_ary = sb*np.logspace(-1, 1, 11, dtype=DTYPE)

    cdef double[:,::1] g1_ref_mat = np.empty((len(sb_ref_ary), k_max + 1), dtype=DTYPE)
    cdef double[:,::1] g2_ref_mat = np.empty((len(sb_ref_ary), k_max + 1), dtype=DTYPE)

    cdef double[:,::1] g1_mat_f = np.empty((npix_roi, k_max+1), dtype=DTYPE)
    cdef double[:,::1] g2_mat_f = np.empty((npix_roi, k_max+1), dtype=DTYPE)

    cdef double[:] g1_ref_ary = np.empty(len(sb_ref_ary), dtype=DTYPE)
    cdef double[:] g2_ref_ary = np.empty(len(sb_ref_ary), dtype=DTYPE)

    cdef double[:] g1_int_ary = np.empty(npix_roi, dtype=DTYPE)
    cdef double[:] g2_int_ary = np.empty(npix_roi, dtype=DTYPE)

    cdef double[::1] ffac_ary = np.empty(npix_roi, dtype=DTYPE)
    cdef double[:] sb_eff_ary = np.empty (npix_roi, dtype=DTYPE)
    cdef double[:,::1] sb_eff_mat = np.zeros((npix_roi, k_max+1), dtype=DTYPE)

    cdef bool ft_cond = False

    if len(ft_compressed) != 0:
        ft_cond = True
        # print("Nontrivial flux template provided")
        # assert(0 not in ft_compressed), \
        #     "Flux template contains one or more 0's!"       
    # if ft_cond != True:
    #     print("No nontrivial flux templates")

    for f_index in range(len(f_ary)):
        f2 = float(f_ary[f_index])
        df_rho_div_f2 = df_rho_div_f_ary[f_index]

        # For the case of a nontrivial flux template, generate matrix of incomplete gamma function values
        # to interpolate about, and interpolate appropriately for each pixel
        if ft_cond:
            for i in range(len(sb_ref_ary)):
                g1_ary_f = igf.incgamma_up_fct_ary(k_max, 1. - n1, sb_ref_ary[i] * f2)
                g1_ref_mat[i,:] = g1_ary_f
                g2_ary_f = igf.incgamma_lo_fct_ary(k_max, 1. - n2, sb_ref_ary[i] * f2)
                g2_ref_mat[i,:] = g2_ary_f
            
            for p in range(npix_roi):
                for k in range(k_max):
                    if k <= data[p]+1:
                        sb_eff_mat[p,k] = ft_compressed[p] * sb 

            for k in range(k_max):
                g1_ref_ary = g1_ref_mat[:,k]
                g2_ref_ary = g2_ref_mat[:,k]
                sb_eff_ary = sb_eff_mat[:,k]
                # print("sb_eff_ary = ", sb_eff_ary)

                minval1 = findmin.minval(g1_ref_ary)
                if minval1 <= 0:
                    g1_int_ary = interp.logloginterp1d(sb_ref_ary, g1_ref_ary, sb_eff_ary, offset = 1-minval1) 
                else:
                    g1_int_ary = interp.logloginterp1d(sb_ref_ary, g1_ref_ary, sb_eff_ary)
                g1_mat_f[:,k] = g1_int_ary
                # print("g1_int_ary = ", g1_int_ary)

                minval2 = findmin.minval(g2_ref_ary)
                if minval2 <= 0:
                    g2_int_ary = interp.logloginterp1d(sb_ref_ary, g2_ref_ary, sb_eff_ary, offset = 1-minval2) 
                else:
                    g2_int_ary = interp.logloginterp1d(sb_ref_ary, g2_ref_ary, sb_eff_ary)
                g2_mat_f[:,k] = g2_int_ary
                # print("g2_int_ary = ", np.asarray(g2_int_ary))


            for p in range(npix_roi):
                a_ps = float(theta[0])/ft_compressed[p]
                sb = ft_compressed[p] * float(theta[3])

                pref1_x_m_ary =  pow(sb * f2, n1)
                pref2_x_m_ary = pow(sb * f2, n2)

                x_m_sum_f = a_ps*sb*f2*(1/(n1-1)+1/(1-n2)) * npt_compressed[p]
                x_m_sum[p] += df_rho_div_f2*x_m_sum_f

                for k in range(data[p]+1):
                    # print('g1=', g1_mat_f[p])
                    # print('g2=',g2_mat_f[p])
                    x_m_ary_f = a_ps * (pref1_x_m_ary*g1_mat_f[p,k] + pref2_x_m_ary*g2_mat_f[p,k]) * npt_compressed[p]
                    x_m_ary[p,k] += df_rho_div_f2*x_m_ary_f

        else:
            g1_ary_f = igf.incgamma_up_fct_ary(k_max, 1. - n1, sb * f2)
            g2_ary_f = igf.incgamma_lo_fct_ary(k_max, 1. - n2, sb * f2)

            for p in range(npix_roi):
                pref1_x_m_ary =  pow(sb * f2, n1)
                pref2_x_m_ary = pow(sb * f2, n2)

                x_m_sum_f = a_ps*sb*f2*(1/(n1-1)+1/(1-n2)) * npt_compressed[p]
                x_m_sum[p] += df_rho_div_f2*x_m_sum_f

                for k in range(data[p]+1):
                    x_m_ary_f = a_ps * (pref1_x_m_ary*g1_ary_f[k] + pref2_x_m_ary*g2_ary_f[k]) * npt_compressed[p]
                    x_m_ary[p,k] += df_rho_div_f2*x_m_ary_f
            
    x_m_sum = np.asarray(x_m_sum) - np.asarray(x_m_ary)[:,0] 

    # return np.asarray(g1_ref_mat), np.asarray(g2_ref_mat)
    return np.asarray(x_m_ary), np.asarray(x_m_sum)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
def return_xs_2break(double[::1] theta, double[::1] f_ary, 
                     double[::1] df_rho_div_f_ary, double[::1] ft_compressed, double[::1] npt_compressed,
                     int[::1] data):
    """ Dedicated calculation of x_m and x_m_sum for 2 breaks 
    """

    cdef float a_ps = float(theta[0])
    cdef float n1 = float(theta[1])
    cdef float n2 = float(theta[2])
    cdef float n3 = float(theta[3])
    cdef float sb1 = float(theta[4])
    cdef float sb2 = float(theta[5])

    cdef int k_max = int(np.max(data) + 1)
    cdef int npix_roi = len(npt_compressed)

    cdef double[:,::1] x_m_ary = np.zeros(shape=(npix_roi,k_max + 1))
    cdef double[::1] x_m_sum = np.zeros(npix_roi)

    cdef double[::1] g0_ary_f = np.zeros(k_max + 1)
    cdef double[::1] g1_ary_f = np.zeros(k_max + 1)
    cdef double[::1] g1_ary_f_1 = np.zeros(k_max + 1)
    cdef double[::1] g1_ary_f_2 = np.zeros(k_max + 1)
    cdef double[::1] g2_ary_f = np.zeros(k_max + 1)

    cdef double f2, df_rho_div_f2
    cdef double pref0_x_m_ary, pref1_x_m_ary, pref2_x_m_ary
    cdef double first0_x_m_sum_ary, first1_x_m_sum_ary, first2_x_m_sum_ary
    cdef double second0_x_m_sum_ary, second1_x_m_sum_ary, second2_x_m_sum_ary
    cdef double x_m_sum_f, x_m_ary_f
    cdef double minval0, minval1, minval2

    cdef Py_ssize_t f_index, p, k, i, j

    first0_x_m_sum_ary = 1/(1-n1)
    first1_x_m_sum_ary = 1/(1-n2)
    first2_x_m_sum_ary = 1/(1-n3)
 
    second0_x_m_sum_ary = -1.0
    second1_x_m_sum_ary = (1 - pow(sb2/sb1, 1-n2))
    second2_x_m_sum_ary = pow(sb2/sb1, 1-n2)

    cdef double[:] sb1_ref_ary = sb1*np.logspace(-1, 1, 11, dtype=DTYPE)
    cdef double[:] sb2_ref_ary = sb2*np.logspace(-1, 1, 11, dtype=DTYPE)

    cdef double[:,::1] g0_ref_mat = np.empty((len(sb1_ref_ary), k_max + 1), dtype=DTYPE)
    cdef double[:,::1] g1_ref_mat_1 = np.empty((len(sb1_ref_ary), k_max + 1), dtype=DTYPE)
    cdef double[:,::1] g1_ref_mat_2 = np.empty((len(sb2_ref_ary), k_max + 1), dtype=DTYPE)
    cdef double[:,::1] g2_ref_mat = np.empty((len(sb2_ref_ary), k_max + 1), dtype=DTYPE)

    cdef double[:,::1] g0_mat_f = np.empty((npix_roi, k_max+1), dtype=DTYPE)    
    cdef double[:,::1] g1_mat_f_1 = np.empty((npix_roi, k_max+1), dtype=DTYPE)
    cdef double[:,::1] g1_mat_f_2 = np.empty((npix_roi, k_max+1), dtype=DTYPE)
    cdef double[:,::1] g2_mat_f = np.empty((npix_roi, k_max+1), dtype=DTYPE)

    cdef double[:] g0_ref_ary = np.empty(len(sb1_ref_ary), dtype=DTYPE)
    cdef double[:] g1_ref_ary_1 = np.empty(len(sb1_ref_ary), dtype=DTYPE)
    cdef double[:] g1_ref_ary_2 = np.empty(len(sb2_ref_ary), dtype=DTYPE)
    cdef double[:] g2_ref_ary = np.empty(len(sb2_ref_ary), dtype=DTYPE)

    cdef double[:] g0_int_ary = np.empty(npix_roi, dtype=DTYPE)
    cdef double[:] g1_int_ary_1 = np.empty(npix_roi, dtype=DTYPE)
    cdef double[:] g1_int_ary_2 = np.empty(npix_roi, dtype=DTYPE)
    cdef double[:] g2_int_ary = np.empty(npix_roi, dtype=DTYPE)

    cdef double[::1] ffac_ary = np.empty(npix_roi, dtype=DTYPE)
    cdef double[:] sb1_eff_ary = np.empty (npix_roi, dtype=DTYPE)
    cdef double[:] sb2_eff_ary = np.empty (npix_roi, dtype=DTYPE)
    cdef double[:,::1] sb1_eff_mat = np.zeros((npix_roi, k_max+1), dtype=DTYPE)
    cdef double[:,::1] sb2_eff_mat = np.zeros((npix_roi, k_max+1), dtype=DTYPE)

    cdef bool ft_cond = False

    if len(ft_compressed) != 0:
        ft_cond = True
        # print("Nontrivial flux template provided")
        # assert(0 not in ft_compressed), \
        #     "Flux template contains one or more 0's!"       
    # if ft_cond != True:
    #     print("No nontrivial flux templates")

    for f_index in range(len(f_ary)):
        f2 = float(f_ary[f_index])
        df_rho_div_f2 = df_rho_div_f_ary[f_index]

        if ft_cond:
            for i in range(len(sb1_ref_ary)):
                g0_ary_f = igf.incgamma_up_fct_ary(k_max, 1. - n1, sb1_ref_ary[i] * f2)
                g0_ref_mat[i,:] = g0_ary_f
                # g1_ary_f_1 = igf.incgamma_up_fct_ary(k_max, 1. - n2, sb1_ref_ary[i] * f2)
                # g1_ref_mat_1[i,:] = g1_ary_f_1

            for j in range(len(sb2_ref_ary)):
                g1_ary_f_2 = igf.incgamma_up_fct_ary(k_max, 1. - n2, sb2_ref_ary[j] * f2) 
                g1_ref_mat_2[j,:] = g1_ary_f_2
                # g2_ary_f = igf.incgamma_lo_fct_ary(k_max, 1. - n3, sb2_ref_ary[j] * f2)
                # g2_ref_mat[j,:] = g2_ary_f
            
            for p in range(npix_roi):
                for k in range(k_max):
                    if k <= data[p]+1:
                        sb1_eff_mat[p,k] = ft_compressed[p] * sb1
                        sb2_eff_mat[p,k] = ft_compressed[p] * sb2

            for k in range(k_max):
                g0_ref_ary = g0_ref_mat[:,k]
                # g1_ref_ary_1 = g1_ref_mat_1[:,k]
                # g1_ref_ary_2 = g1_ref_mat_2[:,k]
                g2_ref_ary = g2_ref_mat[:,k]
                sb1_eff_ary = sb1_eff_mat[:,k]
                sb2_eff_ary = sb2_eff_mat[:,k]

                minval0 = findmin.minval(g0_ref_ary)
                if minval0 <= 0:
                    g0_int_ary = interp.logloginterp1d(sb1_ref_ary, g0_ref_ary, sb1_eff_ary, offset = 1-minval0) 
                else:
                    g0_int_ary = interp.logloginterp1d(sb1_ref_ary, g0_ref_ary, sb1_eff_ary)
                g0_mat_f[:,k] = g0_int_ary

                # minval1_1 = findmin.minval(g1_ref_ary_1)
                # if minval1_1 <= 0:
                #     g1_int_ary_1 = interp.logloginterp1d(sb1_ref_ary, g1_ref_ary_1, sb1_eff_ary, offset = 1-minval1_1) 
                # else:
                #     g1_int_ary_1 = interp.logloginterp1d(sb1_ref_ary, g1_ref_ary_1, sb1_eff_ary)
                # g1_mat_f_1[:,k] = g1_int_ary_1

                # minval1_2 = findmin.minval(g1_ref_ary_2)
                # if minval1_2 <= 0:
                #     g1_int_ary_2 = interp.logloginterp1d(sb2_ref_ary, g1_ref_ary_2, sb2_eff_ary, offset = 1-minval1_2) 
                # else:
                #     g1_int_ary_2 = interp.logloginterp1d(sb2_ref_ary, g1_ref_ary_2, sb2_eff_ary)
                # g1_mat_f_2[:,k] = g1_int_ary_2

                minval2 = findmin.minval(g2_ref_ary)
                if minval2 <= 0:
                    g2_int_ary = interp.logloginterp1d(sb2_ref_ary, g2_ref_ary, sb2_eff_ary, offset = 1-minval2) 
                else:
                    g2_int_ary = interp.logloginterp1d(sb2_ref_ary, g2_ref_ary, sb2_eff_ary)
                g2_mat_f[:,k] = g2_int_ary

            for p in range(npix_roi):
                a_ps = float(theta[0])/ft_compressed[p]
                sb1 = ft_compressed[p] * float(theta[4])
                sb2 = ft_compressed[p] * float(theta[5])
             
                second1_x_m_sum_ary = (1 - pow(sb2/sb1, 1-n2))
                second2_x_m_sum_ary = pow(sb2/sb1, 1-n2)

                pref0_x_m_ary = pow(sb1 * f2, n1)
                pref1_x_m_ary = pref0_x_m_ary * pow(sb1 * f2, n2 - n1)
                pref2_x_m_ary = pref1_x_m_ary * pow(sb2 * f2, n3 - n2)

                x_m_sum_f = a_ps * (sb1 * f2) * (first0_x_m_sum_ary*second0_x_m_sum_ary
                            + first1_x_m_sum_ary*second1_x_m_sum_ary
                            + first2_x_m_sum_ary*second2_x_m_sum_ary) \
                            * npt_compressed[p]
                x_m_sum[p] += df_rho_div_f2 * x_m_sum_f

                for k in range(data[p]+1):
                    x_m_ary_f = a_ps * (pref0_x_m_ary * g0_mat_f[p,k] 
                                + pref1_x_m_ary * (g1_ary_f[k])
                                + pref2_x_m_ary * g2_mat_f[p,k]) * npt_compressed[p]
                    # x_m_ary_f = a_ps * (pref0_x_m_ary * g0_mat_f[p,k] 
                    #             + pref1_x_m_ary * (g1_mat_f_2[p,k] - g1_mat_f_1[p,k])
                    #             + pref2_x_m_ary * g2_mat_f[p,k]) * npt_compressed[p]
                    x_m_ary[p,k] += df_rho_div_f2 * x_m_ary_f

        else:
            g0_ary_f = igf.incgamma_up_fct_ary(k_max, 1. - n1, sb1 * f2)
            g1_ary_f = igf.incgamma_up_fct_ary(k_max, 1. - n2, sb2 * f2) \
                       - igf.incgamma_up_fct_ary(k_max, 1. - n2, sb1 * f2)
            g2_ary_f = igf.incgamma_lo_fct_ary(k_max, 1. - n3, sb2 * f2)

            pref0_x_m_ary = pow(sb1 * f2, n1)
            pref1_x_m_ary = pref0_x_m_ary * pow(sb1 * f2, n2 - n1)
            pref2_x_m_ary = pref1_x_m_ary * pow(sb2 * f2, n3 - n2)

            for p in range(npix_roi):
                x_m_sum_f = a_ps * (sb1 * f2) * (first0_x_m_sum_ary*second0_x_m_sum_ary
                            + first1_x_m_sum_ary*second1_x_m_sum_ary
                            + first2_x_m_sum_ary*second2_x_m_sum_ary) \
                            * npt_compressed[p]
                x_m_sum[p] += df_rho_div_f2 * x_m_sum_f

                for k in range(data[p]+1):
                    x_m_ary_f = a_ps * (pref0_x_m_ary * g0_ary_f[k] + pref1_x_m_ary
                                * g1_ary_f[k] + pref2_x_m_ary * g2_ary_f[k]) \
                                * npt_compressed[p]
                    x_m_ary[p,k] += df_rho_div_f2 * x_m_ary_f

    x_m_sum = np.asarray(x_m_sum) - np.asarray(x_m_ary)[:,0] 

    return np.asarray(x_m_ary), np.asarray(x_m_sum)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
def return_xs_2break_orig(double[::1] theta, double[::1] f_ary, 
                     double[::1] df_rho_div_f_ary, double[::1] ft_compressed, double[::1] npt_compressed,
                     int[::1] data):
    """ Dedicated calculation of x_m and x_m_sum for 2 breaks 
    """

    cdef float a_ps = float(theta[0])
    cdef float n1 = float(theta[1])
    cdef float n2 = float(theta[2])
    cdef float n3 = float(theta[3])
    cdef float sb1 = float(theta[4])
    cdef float sb2 = float(theta[5])

    cdef int k_max = int(np.max(data) + 1)
    cdef int npix_roi = len(npt_compressed)

    cdef double[:,::1] x_m_ary = np.zeros(shape=(npix_roi,k_max + 1))
    cdef double[::1] x_m_sum = np.zeros(npix_roi)

    cdef double[::1] g0_ary_f = np.zeros(k_max + 1)
    cdef double[::1] g1_ary_f = np.zeros(k_max + 1)
    cdef double[::1] g2_ary_f = np.zeros(k_max + 1)

    cdef double f2, df_rho_div_f2
    cdef double pref0_x_m_ary, pref1_x_m_ary, pref2_x_m_ary
    cdef double first0_x_m_sum_ary, first1_x_m_sum_ary, first2_x_m_sum_ary
    cdef double second0_x_m_sum_ary, second1_x_m_sum_ary, second2_x_m_sum_ary
    cdef double x_m_sum_f, x_m_ary_f

    cdef Py_ssize_t f_index, p, k

    first0_x_m_sum_ary = 1/(1-n1)
    first1_x_m_sum_ary = 1/(1-n2)
    first2_x_m_sum_ary = 1/(1-n3)

    second0_x_m_sum_ary = -1.0
    second1_x_m_sum_ary = (1 - pow(sb2/sb1, 1-n2))
    second2_x_m_sum_ary = pow(sb2/sb1, 1-n2)

    for f_index in range(len(f_ary)):
        f2 = float(f_ary[f_index])
        df_rho_div_f2 = df_rho_div_f_ary[f_index]
        
        g0_ary_f = igf.incgamma_up_fct_ary(k_max, 1. - n1, sb1 * f2)
        g1_ary_f = igf.incgamma_up_fct_ary(k_max, 1. - n2, sb2 * f2) \
                   - igf.incgamma_up_fct_ary(k_max, 1. - n2, sb1 * f2)
        g2_ary_f = igf.incgamma_lo_fct_ary(k_max, 1. - n3, sb2 * f2)

        pref0_x_m_ary = pow(sb1 * f2, n1)
        pref1_x_m_ary = pref0_x_m_ary * pow(sb1 * f2, n2 - n1)
        pref2_x_m_ary = pref1_x_m_ary * pow(sb2 * f2, n3 - n2)

        for p in range(npix_roi):
            x_m_sum_f = a_ps * (sb1 * f2) * (first0_x_m_sum_ary*second0_x_m_sum_ary
                        + first1_x_m_sum_ary*second1_x_m_sum_ary
                        + first2_x_m_sum_ary*second2_x_m_sum_ary) \
                        * npt_compressed[p]
            x_m_sum[p] += df_rho_div_f2 * x_m_sum_f

            for k in range(data[p]+1):
                x_m_ary_f = a_ps * (pref0_x_m_ary * g0_ary_f[k] + pref1_x_m_ary
                            * g1_ary_f[k] + pref2_x_m_ary * g2_ary_f[k]) \
                            * npt_compressed[p]
                x_m_ary[p,k] += df_rho_div_f2 * x_m_ary_f

    x_m_sum = np.asarray(x_m_sum) - np.asarray(x_m_ary)[:,0] 

    return np.asarray(x_m_ary), np.asarray(x_m_sum)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
def return_xs_3break(double[::1] theta, double[::1] f_ary, double[::1] df_rho_div_f_ary, 
                     double[::1] ft_compressed, double[::1] npt_compressed, int[::1] data):
    """ Dedicated calculation of x_m and x_m_sum for 3 breaks 
    """

    cdef float a_ps = float(theta[0])
    cdef float n1 = float(theta[1])
    cdef float n2 = float(theta[2])
    cdef float n3 = float(theta[3])
    cdef float n4 = float(theta[4])
    cdef float sb1 = float(theta[5])
    cdef float sb2 = float(theta[6])
    cdef float sb3 = float(theta[7])

    cdef int k_max = int(np.max(data) + 1)
    cdef int npix_roi = len(npt_compressed)

    cdef double[:,::1] x_m_ary = np.zeros((npix_roi,k_max + 1))
    cdef double[::1] x_m_sum = np.zeros(npix_roi)

    cdef double[::1] g0_ary_f = np.zeros(k_max + 1)
    cdef double[::1] g1_ary_f = np.zeros(k_max + 1)
    cdef double[::1] g2_ary_f = np.zeros(k_max + 1)
    cdef double[::1] g3_ary_f = np.zeros(k_max + 1)

    cdef double f2, df_rho_div_f2, fluxfac
    cdef double pref0_x_m_ary, pref1_x_m_ary, pref2_x_m_ary, pref3_x_m_ary
    cdef double first0_x_m_sum_ary, first1_x_m_sum_ary, first2_x_m_sum_ary
    cdef double first3_x_m_sum_ary
    cdef double second0_x_m_sum_ary, second1_x_m_sum_ary, second2_x_m_sum_ary
    cdef double second3_x_m_sum_ary
    cdef double x_m_sum_f, x_m_ary_f

    cdef Py_ssize_t f_index, p, k

    first0_x_m_sum_ary = 1/(1-n1)
    first1_x_m_sum_ary = 1/(1-n2)
    first2_x_m_sum_ary = 1/(1-n3)
    first3_x_m_sum_ary = 1/(1-n4)

    second0_x_m_sum_ary = -1.0
    second1_x_m_sum_ary = (1 - pow(sb2/sb1, 1-n2))
    second2_x_m_sum_ary = pow(sb2/sb1, 1-n2) * (1 - pow(sb3/sb2, 1-n3))
    second3_x_m_sum_ary = pow(sb2/sb1, 1-n2) * pow(sb3/sb2, 1-n3)

    for f_index in range(len(f_ary)):
        f2 = float(f_ary[f_index])
        df_rho_div_f2 = df_rho_div_f_ary[f_index]
        g0_ary_f = igf.incgamma_up_fct_ary(k_max, 1. - n1, sb1 * f2)
        g1_ary_f = igf.incgamma_up_fct_ary(k_max, 1. - n2, sb2 * f2) \
                   - igf.incgamma_up_fct_ary(k_max, 1. - n2, sb1 * f2)
        g2_ary_f = igf.incgamma_up_fct_ary(k_max, 1. - n3, sb3 * f2) \
                   - igf.incgamma_up_fct_ary(k_max, 1. - n3, sb2 * f2)
        g3_ary_f = igf.incgamma_lo_fct_ary(k_max, 1. - n4, sb3 * f2)

        for p in range(npix_roi):
            fluxfac = ft_compressed[p]
            if fluxfac != 1:
                g0_ary_f = igf.incgamma_up_fct_ary(k_max, 1. - n1, sb1 * fluxfac * f2)
                g1_ary_f = igf.incgamma_up_fct_ary(k_max, 1. - n2, sb2 * fluxfac * f2) \
                           - igf.incgamma_up_fct_ary(k_max, 1. - n2, sb1 * fluxfac * f2)
                g2_ary_f = igf.incgamma_up_fct_ary(k_max, 1. - n3, sb3 * fluxfac * f2) \
                           - igf.incgamma_up_fct_ary(k_max, 1. - n3, sb2 * fluxfac * f2)
                g3_ary_f = igf.incgamma_lo_fct_ary(k_max, 1. - n4, sb3 * fluxfac * f2)

            pref0_x_m_ary = pow(sb1 * fluxfac * f2, n1)
            pref1_x_m_ary = pref0_x_m_ary * pow(sb1 * fluxfac * f2, n2 - n1)
            pref2_x_m_ary = pref1_x_m_ary * pow(sb2 * fluxfac * f2, n3 - n2)
            pref3_x_m_ary = pref2_x_m_ary * pow(sb3 * fluxfac * f2, n4 - n3)

            x_m_sum_f = (a_ps/fluxfac * sb1 * fluxfac * f2) * (first0_x_m_sum_ary*second0_x_m_sum_ary
                        + first1_x_m_sum_ary*second1_x_m_sum_ary
                        + first2_x_m_sum_ary*second2_x_m_sum_ary
                        + first3_x_m_sum_ary*second3_x_m_sum_ary) \
                        * npt_compressed[p]
            x_m_sum[p] += df_rho_div_f2 * x_m_sum_f

            for k in range(data[p]+1):
                x_m_ary_f = a_ps/fluxfac * (pref0_x_m_ary*g0_ary_f[k]
                            + pref1_x_m_ary*g1_ary_f[k]
                            + pref2_x_m_ary*g2_ary_f[k]
                            + pref3_x_m_ary*g3_ary_f[k]) \
                            * npt_compressed[p]
                x_m_ary[p,k] += df_rho_div_f2 * x_m_ary_f

    x_m_sum = np.asarray(x_m_sum) - np.asarray(x_m_ary)[:,0] 

    return np.asarray(x_m_ary), np.asarray(x_m_sum)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
@cython.initializedcheck(False)
def return_xs_lbreak(double[::1] theta, int n_break, double[::1] f_ary, 
                     double[::1] df_rho_div_f_ary, double[::1] npt_compressed,
                     int[::1] data):
    """ General calculation of x_m and x_m_sum for l breaks 
    """

    cdef float a_ps = float(theta[0])
    cdef double[::1] n_ary = theta[1:n_break + 2]
    cdef double[::1] sb_ary = theta[n_break + 2: 2*n_break + 2]

    cdef int k_max = int(np.max(data) + 1)
    cdef int npix_roi = len(npt_compressed)

    cdef double[:,::1] x_m_ary = np.zeros((npix_roi, k_max + 1))
    cdef double[::1] x_m_sum = np.zeros(npix_roi)

    cdef double[:,::1] g_ary_f_ary = np.zeros((n_break + 1, k_max + 1))
    cdef double[::1] pref_x_m_ary = np.zeros(n_break + 1)

    cdef double[::1] first_x_m_sum_ary = np.zeros(n_break + 1)
    cdef double[::1] second_x_m_sum_ary = np.zeros(n_break + 1)

    cdef double[::1] gamma_ary = np.zeros(k_max + 1)

    cdef double f2, df_rho_div_f2

    cdef double[::1] sbpow_ary = np.zeros(n_break + 1)

    cdef double dp = 0.0, dp_sum = 0

    cdef double x_m_sum_f, x_m_ary_f

    cdef Py_ssize_t f_index, p, k, m, i, j

    # Get (Sb[j]/Sb[j+1])**(-n[j]) to avoid having to call pow() repeatedly 
    for j in range(n_break + 1):
        sbpow_ary[j] = pow(sb_ary[j] / sb_ary[j-1], - n_ary[j])

    # x_m_sum factors 1
    first_x_m_sum_ary[0] = -1. 

    for i in range(1, n_break + 1):
        first_x_m_sum_ary[i] = first_x_m_sum_ary[0]
        for j in range(1, i):
            first_x_m_sum_ary[i] *= pow(sb_ary[j] / sb_ary[j-1], - n_ary[j])
        first_x_m_sum_ary[i] *= sb_ary[i-1] * ( - 1 + sbpow_ary[i]
                                * (sb_ary[i] / sb_ary[i-1])) / sb_ary[0]

    first_x_m_sum_ary[n_break] = 1.
    for j in range(1, n_break):
        first_x_m_sum_ary[n_break] *= sbpow_ary[j]
    first_x_m_sum_ary[n_break]  *=  sb_ary[n_break-1] / sb_ary[0]

    # x_m_sum factors 2
    for i in range(n_break+1):
        second_x_m_sum_ary[i] = 1/(1 - n_ary[i])

    for f_index in range(len(f_ary)): # PSF loop
        f2 = float(f_ary[f_index])
        df_rho_div_f2 = df_rho_div_f_ary[f_index]

        # Terms involving incomplete gamma functions in x_m
        gamma_ary = igf.incgamma_up_fct_ary(k_max, 1.-n_ary[0], sb_ary[0] * f2)
        for m in range(k_max + 1):
            g_ary_f_ary[0][m] = gamma_ary[m]

        for i in range(1, n_break):
            gamma_ary = igf.incgamma_up_fct_ary(k_max, 1.-n_ary[i], sb_ary[i] * f2) \
                        - igf.incgamma_up_fct_ary(k_max, 1.-n_ary[i], sb_ary[i-1] * f2)
            for m in range(k_max + 1):
                g_ary_f_ary[i][m] = gamma_ary[m]

        gamma_ary = igf.incgamma_lo_fct_ary(k_max, 1.-n_ary[n_break], sb_ary[n_break-1] * f2)
        for m in range(k_max + 1):
            g_ary_f_ary[n_break][m] = gamma_ary[m]

        # Terms not involving incomplete gamma functions in x_m
        pref_x_m_ary[0] = pow(sb_ary[0] * f2, n_ary[0])

        for i in range(1, n_break + 1):
            pref_x_m_ary[i] = pref_x_m_ary[i-1] * pow(sb_ary[i-1] * f2, n_ary[i]-n_ary[i-1])

        dp_sum = 0.0
        for i in range(0, n_break+1):
            dp_sum += first_x_m_sum_ary[i] *second_x_m_sum_ary[i]

        for p in range(npix_roi):
            x_m_sum_f = a_ps * sb_ary[0] * f2 * dp_sum * npt_compressed[p]
            x_m_sum[p] += df_rho_div_f2 * x_m_sum_f
            
            for k in range(data[p]+1):
                dp = 0.0
                for i in range(0, n_break+1):
                    dp += pref_x_m_ary[i] * g_ary_f_ary[i][k]

                x_m_ary_f = a_ps * dp * npt_compressed[p]
                x_m_ary[p,k] += df_rho_div_f2 * x_m_ary_f

    x_m_sum = np.asarray(x_m_sum) - np.asarray(x_m_ary)[:,0] 

    return np.asarray(x_m_ary), np.asarray(x_m_sum)

