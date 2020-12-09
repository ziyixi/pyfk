#!python
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, linetrace=True
# note even we have linetrace=True, it still need to be enabled by
# define_macros=[("CYTHON_TRACE_NOGIL", "1")]
import numpy as np
from libc.math cimport fabs

cpdef taup(const int src_lay_input, const int rcv_lay_input, const double[:] thickness, const double[:] velocity, const double[:] receiver_distance):
    # * init some values
    cdef int num_lay = len(velocity)
    if src_lay_input < rcv_lay_input:
        src_lay, rcv_lay = rcv_lay_input, src_lay_input
    else:
        src_lay, rcv_lay = src_lay_input, rcv_lay_input

    # * wave number root
    # cdef double[:] p2 = 1./velocity**2
    cdef double[:] p2 = np.zeros(len(velocity), dtype=np.float64)
    cdef int ivelocity
    for ivelocity in range(len(velocity)):
        p2[ivelocity] = 1. / velocity[ivelocity]**2
    cdef double[:] t0 = np.zeros(len(receiver_distance), dtype=np.float64)
    cdef double[:] td = np.zeros(len(receiver_distance), dtype=np.float64)
    cdef double[:] p0 = np.zeros(len(receiver_distance), dtype=np.float64)
    cdef double[:] pd = np.zeros(len(receiver_distance), dtype=np.float64)

    # * declare variables in the for loop
    cdef double[:] ray_len
    cdef int topp, bttm
    cdef double distance, min_p2, p, t

    # * the main loop
    cdef int irec
    for irec in range(len(receiver_distance)):
        distance = receiver_distance[irec]
        ray_len = np.zeros(len(thickness), dtype=np.float64)
        topp = rcv_lay + 1
        bttm = src_lay

        # * consider direct wave
        ray_len[(topp - 1):bttm] = thickness[(topp - 1):bttm]
        min_p2 = np.min(p2[(topp - 1):bttm])
        # set min_p as initial gauss of p0, the ray parameter
        pd[irec] = findp0(distance, min_p2**0.5,
                          topp, bttm, ray_len, p2)
        td[irec] = travel(distance, pd[irec], topp, bttm, ray_len, p2)
        # use td,pd as current p0,t0 (starting time)
        t0[irec], p0[irec] = td[irec], pd[irec]

        # * consider reflected wave from below
        topp = rcv_lay + 1
        for bttm in range(src_lay + 1, num_lay):
            ray_len[bttm - 1] = 2. * thickness[bttm - 1]
            if min_p2 > p2[bttm - 1]:
                min_p2 = p2[bttm - 1]
            p = findp0(distance, min_p2**0.5,
                       topp, bttm, ray_len, p2)
            if min_p2 > p2[bttm]:
                min_p2 = p2[bttm]
            if p > min_p2**0.5:
                p = min_p2**0.5
            # t for current assumed layers
            t = travel(distance, p, topp, bttm, ray_len, p2)
            if t < t0[irec]:
                t0[irec], p0[irec] = t, p

        # * consider reflected wave from above
        bttm = src_lay
        for topp in range(rcv_lay, 0, -1):
            ray_len[topp - 1] = 2. * thickness[topp - 1]
            # the strange thing is that whether we should reset min_p2, maybe
            # also consider sSMS?
            if min_p2 > p2[topp - 1]:
                min_p2 = p2[topp - 1]
            p = findp0(distance, min_p2**0.5,
                       topp, bttm, ray_len, p2)
            if (topp > 1) and (min_p2 > p2[topp - 2]):
                min_p2 = p2[topp - 2]
            if p > min_p2**0.5:
                p = min_p2**0.5
            t = travel(distance, p, topp, bttm, ray_len, p2)
            if t < t0[irec]:
                t0[irec], p0[irec] = t, p
    return np.asarray(t0), np.asarray(td), np.asarray(p0), np.asarray(pd)

cdef double findp0(double distance, double p0_gauss, int topp, int bttm, double[:] ray_len, double[:] p2):
    cdef double p1_search = 0, p2_search = 0
    while p1_search != p0_gauss:
        p2_search = p0_gauss
        p0_gauss = (p1_search + p2_search) / 2
        dtdp_gauss = dtdp(distance, p0_gauss, topp, bttm, ray_len, p2)
        if (fabs(dtdp_gauss) < 1.e-7) or (p0_gauss ==
                                          p1_search) or (p0_gauss == p2_search):
            return p0_gauss
        if dtdp_gauss > 0:
            p1_search = p0_gauss
            p0_gauss = p2_search
    return p0_gauss

cdef double travel(double distance, double p0_gauss, int topp, int bttm, double[:] ray_len, double[:] p2):
    cdef double result = distance * p0_gauss, p0_gauss2 = p0_gauss**2
    # result += np.sum(np.sqrt(p2[topp-1:bttm]-p0_gauss2)*ray_len[topp-1:bttm])
    cdef int index
    for index in range(topp - 1, bttm):
        result += ((p2[index] - p0_gauss2)**0.5) * ray_len[index]
    return result

cdef double dtdp(double distance, double p0_gauss, int topp, int bttm, double[:] ray_len, double[:] p2):
    cdef double p0_gauss2 = p0_gauss**2, result = 0
    cdef int index
    for index in range(topp - 1, bttm):
        result -= ray_len[index] / (p2[index] - p0_gauss2)**0.5
    result = distance + p0_gauss * result
    return result
