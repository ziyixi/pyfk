#!python
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, linetrace=True
# note even we have linetrace=True, it still need to be enabled by
# define_macros=[("CYTHON_TRACE_NOGIL", "1")]
import numpy as np

from scipy.special.cython_special cimport jv
from libc.math cimport pi, cos, sin, exp
from pyfk.utils.complex cimport clog, csqrt, creal, cimag, conj
from cysignals.signals cimport sig_on, sig_off

IF PYFK_USE_MPI == "1":
    from mpi4py import MPI
    from mpi4py cimport libmpi as mpi

IF PYFK_USE_MPI == "1":
    def mpi_info():
        vendor = MPI.get_vendor()
        version = vendor[0]+" "+".".join(map(str, vendor[1]))
        var = f"MPI installed correctly, version: {version}."
        print(var)
        return var
ELSE:
    def mpi_info():
        var = f"MPI is not used."
        print(var)
        return var


def _waveform_integration(
        nfft2,
        dw,
        pmin,
        dk,
        kc,
        pmax,
        receiver_distance,
        wc1,
        vs,
        vp,
        qs,
        qp,
        flip,
        filter_const,
        dynamic,
        wc2,
        t0,
        src_type,
        taper,
        wc,
        mu,
        thickness,
        si,
        src_layer,
        rcv_layer,
        updn,
        epsilon,
        sigma,
        sum_waveform):
    sig_on()
    _waveform_integration_sigin(
        nfft2,
        dw,
        pmin,
        dk,
        kc,
        pmax,
        receiver_distance,
        wc1,
        vs,
        vp,
        qs,
        qp,
        flip,
        filter_const,
        dynamic,
        wc2,
        t0,
        src_type,
        taper,
        wc,
        mu,
        thickness,
        si,
        src_layer,
        rcv_layer,
        updn,
        epsilon,
        sigma,
        sum_waveform)
    sig_off()


cdef void _waveform_integration_sigin(int nfft2, double dw, double pmin, double dk, double kc, double pmax,
                                      double[:] receiver_distance, int wc1,
                                      double[:] vs, double[:] vp, double[:] qs, double[:] qp,
                                      bint flip, double filter_const, bint dynamic, int wc2, double[:] t0, int src_type,
                                      double taper, int wc, double[:] mu, double[:] thickness, double[:, :] si,
                                      int src_layer, int rcv_layer, int updn, double epsilon, double sigma, double complex[:, :, :] sum_waveform):
    # * init mpi
    # mpi.MPI_Init(NULL,NULL)
    cdef int size = 1, rank = 0
    IF PYFK_USE_MPI == "1":
        cdef mpi.MPI_Comm comm = mpi.MPI_COMM_WORLD
        mpi.MPI_Comm_size(comm, & size)
        mpi.MPI_Comm_rank(comm, & rank)

    cdef:
        int ik, idep, n, i, irec, flip_val, icom
        double ztemp, k, kc2, omega, z, aj0, aj1, aj2, filtering, phi
        double complex w, att, atttemp, nf
    # * init some arrays
    cdef:
        # double complex[::1] kp = np.zeros(len(thickness), dtype=complex)
        # double complex[::1] ks = np.zeros(len(thickness), dtype=complex)
        double complex[:, ::1] u = np.zeros((3, 3), dtype=complex)
        double complex[:, ::1] aaa = np.zeros((5, 5), dtype=complex)
        double complex[:, ::1] bbb = np.zeros((7, 7), dtype=complex)
        double complex[:, ::1] ccc = np.zeros((7, 7), dtype=complex)
        double complex[::1] eee = np.zeros(7, dtype=complex)
        double complex[::1] ggg = np.zeros(7, dtype=complex)
        double complex[:, ::1] zzz = np.zeros((3, 5), dtype=complex)
        double complex[:, ::1] sss = np.zeros((3, 6), dtype=complex)
        double complex[:, ::1] temppp = np.zeros((4, 4), dtype=complex)

        double complex[::1] ggg_temp = np.zeros(7, dtype=complex)
        double complex[:, ::1] zzz_temp = np.zeros((3, 5), dtype=complex)
        double complex[:, ::1] bbb_temp = np.zeros((7, 7), dtype=complex)

    # * get the n list
    cdef:
        int[::1] n_list = np.zeros(nfft2-wc1+1, dtype=np.intc), n_list_accumulate = np.zeros(nfft2-wc1+1, dtype=np.intc)
        int n_total = 0
        # some ik determined value
        double complex[:, ::1] kp_list = np.zeros((nfft2-wc1+1, len(thickness)), dtype=complex)
        double complex[:, ::1] ks_list = np.zeros((nfft2-wc1+1, len(thickness)), dtype=complex)
        int[::1] flip_val_list = np.zeros(nfft2-wc1+1, dtype=np.intc)
        # work for mpi each rank
        double complex[:, ::1] kp_list_each_rank = np.zeros((nfft2-wc1+1, len(thickness)), dtype=complex)
        double complex[:, ::1] ks_list_each_rank = np.zeros((nfft2-wc1+1, len(thickness)), dtype=complex)
        int[::1] n_list_each_rank = np.zeros(nfft2-wc1+1, dtype=np.intc)

    if flip:
        flip_val_list[:] = -1
    else:
        flip_val_list[:] = 1

    for ik in range(wc1 - 1+rank, nfft2, size):
        kc2 = kc**2
        omega = ik * dw
        w = omega - sigma * 1j
        att = clog(w / (2 * pi)) / pi + 0.5j
        k = omega * pmin + 0.5 * dk
        n = int(((kc2 + (pmax * omega)**2)**0.5 - k) / dk)
        n_list_each_rank[ik-wc1+1] = n
        # n_total = n_total+n
        # n_list_accumulate[ik-wc1+1] = n_total
        # fill in values
        for idep in range(len(thickness)):
            kp_list_each_rank[ik-wc1+1, idep] = (w /
                                                 (vp[idep] * (1. + att / qp[idep])))**2
            ks_list_each_rank[ik-wc1+1, idep] = (w /
                                                 (vs[idep] * (1. + att / qs[idep])))**2
    # reduce to all
    IF PYFK_USE_MPI == "1":
        mpi.MPI_Allreduce(& kp_list_each_rank[0, 0], & kp_list[0, 0], (nfft2-wc1+1)*len(thickness), mpi.MPI_C_DOUBLE_COMPLEX, mpi.MPI_SUM, comm)
        mpi.MPI_Allreduce(& ks_list_each_rank[0, 0], & ks_list[0, 0], (nfft2-wc1+1)*len(thickness), mpi.MPI_C_DOUBLE_COMPLEX, mpi.MPI_SUM, comm)
        mpi.MPI_Allreduce(& n_list_each_rank[0], & n_list[0], nfft2-wc1+1, mpi.MPI_INT, mpi.MPI_SUM, comm)
    ELSE:
        kp_list[:, :] = kp_list_each_rank[:, :]
        ks_list[:, :] = ks_list_each_rank[:, :]
        n_list[:] = n_list_each_rank[:]

    # * get n_total and n_list_accumulate
    for ik in range(wc1 - 1, nfft2):
        n = n_list[ik-wc1+1]
        n_total = n_total+n
        n_list_accumulate[ik-wc1+1] = n_total

    # * we can run the code in parallel for n_total tasks
    # we run n_total loops, then map back
    cdef:
        Py_ssize_t ibool, isearch

    ik = 0
    i = 0
    for ibool in range(rank, n_total, size):
        # find current index ik and i
        for isearch in range(nfft2-wc1+1):
            if(n_list_accumulate[isearch] > ibool):
                ik = isearch+wc1 - 1
                i = ibool-(n_list_accumulate[isearch]-n_list[isearch])
                break

        omega = ik * dw
        k = omega * pmin + (0.5+i) * dk

        # * run kernel
        kernel(
            k,
            u,
            kp_list[ik-wc1+1, :],
            ks_list[ik-wc1+1, :],
            aaa,
            bbb,
            ccc,
            eee,
            ggg,
            zzz,
            sss,
            temppp,
            mu,
            si,
            thickness,
            src_type,
            src_layer,
            rcv_layer,
            updn,
            epsilon,
            # temp
            ggg_temp,
            zzz_temp,
            bbb_temp)
        for irec in range(len(receiver_distance)):
            z = k * receiver_distance[irec]
            aj0 = jv(0., z)
            aj1 = jv(1., z)
            aj2 = jv(2., z)
            # do the numerical integration here
            sum_waveform[irec, 0, ik] += u[0, 0] * \
                aj0 * flip_val_list[ik-wc1+1]
            sum_waveform[irec, 1, ik] += -u[0, 1] * aj1
            sum_waveform[irec, 2, ik] += -u[0, 2] * aj1

            nf = (u[1, 1] + u[1, 2]) * aj1 / z
            sum_waveform[irec, 3, ik] += u[1, 0] * \
                aj1 * flip_val_list[ik-wc1+1]
            sum_waveform[irec, 4, ik] += u[1, 1] * aj0 - nf
            sum_waveform[irec, 5, ik] += u[1, 2] * aj0 - nf

            nf = 2. * (u[2, 1] + u[2, 2]) * aj2 / z
            sum_waveform[irec, 6, ik] += u[2, 0] * \
                aj2 * flip_val_list[ik-wc1+1]
            sum_waveform[irec, 7, ik] += u[2, 1] * aj1 - nf
            sum_waveform[irec, 8, ik] += u[2, 2] * aj1 - nf

    # * we need to gather sum_waveform together and sum them together
    cdef double complex[:, :, :] sum_waveform_all = np.zeros(
        (len(receiver_distance), 9, nfft2), dtype=complex)

    # mpi.MPI_Reduce(& sum_waveform[0, 0, 0], & sum_waveform_all[0, 0, 0], len(receiver_distance)*9*nfft2, mpi.MPI_C_DOUBLE_COMPLEX,
    #                 mpi.MPI_SUM, 0, comm)
    IF PYFK_USE_MPI == "1":
        mpi.MPI_Allreduce(& sum_waveform[0, 0, 0], & sum_waveform_all[0, 0, 0], len(receiver_distance)*9*nfft2, mpi.MPI_C_DOUBLE_COMPLEX, mpi.MPI_SUM, comm)
    ELSE:
        sum_waveform_all[:, :, :] = sum_waveform[:, :, :]

    for ik in range(wc1 - 1, nfft2):
        omega = ik * dw
        # * for each ik, we apply the filtering and apply the time shift in the frequency domain
        filtering = filter_const
        if dynamic and (ik + 1 > wc):
            filtering = 0.5 * (1. + cos((ik + 1 - wc) * taper)) * filtering
        if dynamic and (ik + 1 < wc2):
            filtering = 0.5 * \
                (1. + cos((wc2 - ik - 1) * pi / (wc2 - wc1))) * filtering
        # in fk's code, only apply atttemp for ncom, here we apply to all, with
        # no difference
        for icom in range(9):
            for irec in range(len(receiver_distance)):
                phi = omega * t0[irec]
                atttemp = filtering * (cos(phi) + sin(phi) * 1j)
                sum_waveform[irec, icom,
                             ik] = sum_waveform_all[irec, icom, ik]*atttemp

    # mpi.MPI_Finalize()

cdef void kernel(double k, double complex[:, ::1] u, double complex[::1] kp, double complex[::1] ks,
                 double complex[:, ::1] aaa, double complex[:, ::1] bbb, double complex[:, ::1] ccc, double complex[::1] eee,
                 double complex[::1] ggg, double complex[:, ::1] zzz, double complex[:, ::1] sss, double complex[:, ::1] temppp,
                 double[:] mu, double[:, :] si, double[:] thickness, int src_type, int src_layer, int rcv_layer,
                 int updn, double epsilon,
                 # temps
                 double complex[::1] ggg_temp, double complex[:, ::1] zzz_temp, double complex[:, ::1] bbb_temp):
    # * the code below is directly modified from the FK code
    cdef Py_ssize_t index, ilayer

    # * Explanations:
    # ? aaa --- 4x4 p-sv Haskell matrix and a(5,5)=exb.
    # ? bbb --- product of compound matrices from the receiver to the surface.
    # ? ccc --- compound matrix.
    # ? eee --- vector, the first 5 members are E0|_{12}^{ij}, ij=12,13,23,24,34;
    # ?       the last two are the 1st column of the 2x2 SH E0 (or unit vector if the top is elastic).
    # ? ggg --- vector containing the Rayleigh and Love denominators. It is initialized in the
    # ?       bottom half-space with  (E^-1)|_{ij}^{12}, ij=12,13,23,24,34, and
    # ?       the 1st row of the 2x2 SH E^-1 (or a unit vector if the bottom is vacume).
    # ? zzz --- z(n,j)=s(i)*X|_{ij}^{12} for p-sv and s(i)*X_5i for sh.

    # * we should initialize all the matrix and vectors
    u[:, :] = 0. + 0.j
    aaa[:, :] = 0. + 0.j
    bbb[:, :] = 0. + 0.j
    ccc[:, :] = 0. + 0.j
    eee[:] = 0. + 0.j
    ggg[:] = 0. + 0.j
    zzz[:, :] = 0. + 0.j
    sss[:, :] = 0. + 0.j
    temppp[:, :] = 0. + 0.j

    # * begin initialB
    # ? Initialize bbb as an unit matrix
    # ? eee = (1 0 0 0 0 1 0) for top halfspace boundary condition;
    # ? ggg = (0 0 0 0 1 0 1) for free bottom boundary condition.
    for index in range(7):
        bbb[index, index] = 1.
    eee[0], eee[5], ggg[4], ggg[6] = 1., 1., 1., 1.

    # * main loop
    # * init values used in the loop
    cdef:
        # for ilayer
        double kd, mu2
        double complex kka, kkb, r, ra, rb, r1
        # else
        double complex Ca, Ya, Xa, Cb, Yb, Xb
        double exa, exb, ex
        double complex r2 = 0.+0.j, r3 = 0.+0.j

    for ilayer in range(len(thickness) - 1, -1, -1):
        kka = kp[ilayer] / (k**2)
        kkb = ks[ilayer] / (k**2)
        r = 2. / kkb
        kd = k * thickness[ilayer]
        mu2 = 2. * mu[ilayer]
        ra = csqrt(1. - kka)
        rb = csqrt(1. - kkb)
        r1 = 1. - 1. / r

        if ilayer == len(thickness) - 1 and thickness[ilayer] < epsilon:
            # * begin initialG
            # ? Initialize the g row-vector. The first 5 elements are the
            # ? inverse(E)|_{ij}^{12}, ij=12,13,23,24,34.
            # ? g14 is omitted because g14 = -g23.
            # ? The last two are the 5th row of E^-1.
            # p-sv, see EQ 33 on ZR/p623, constant omitted.
            initialG(ggg, r, ra, rb, r1, mu2)
            # * end initailG
        elif ilayer == 0 and thickness[ilayer] < epsilon:
            # * begin eVector(e)
            # ? The first 5 members are E|_12^ij, ij=12,13,23,24,34.
            # ? The last two are the first column of SH E matrix.
            # For p-sv, compute E|_(12)^(ij), ij=12, 13, 23, 24, 34.
            eVector(eee, ra, rb, r1, mu2)
            # * end eVector(e)
            break
        else:
            # * begin compoundMatrix(c)
            # ? The upper-left 5x5 is the 6x6 compound matrix of the P-SV Haskell matrix,
            # ?       a(ij,kl) = A|_kl^ij, ij=12,13,14,23,24,34,
            # ? after dropping the 3rd row and colummn and replacing the 4th row
            # ? by (2A41, 2A42, 2A44-1,2A45,2A46) (see W&H, P1035).
            # ? The lower-right c 2x2 is the SH part of the Haskell matrix.
            # ? Input: layer parameters.
            # ? Output: compound matrix a, scaled by exa*exb for the P-SV and exb for the SH.
            Ca, Ya, Xa, exa = sh_ch(ra, kd)
            Cb, Yb, Xb, exb = sh_ch(rb, kd)

            compoundMatrix(ccc, Ca, Ya, Xa, Cb, Yb, Xb,
                           exa, exb, r, r1, r2, r3, mu2)
            # * end compoundMatrix(c)

            # * begin propagateG(c, g)
            # ? propagate g vector upward using the compound matrix
            # ?       g = g*a
            propagateG(ggg, ccc, ggg_temp)
            # * end propagateG(c, g)
        if ilayer == src_layer:
            # * begin separatS
            separatS(sss, updn, src_type, ra, rb, r, mu2, temppp, r1, si)
            # * end separatS

            # * begin initialZ(ss, g, z)
            # ? initialize the row-vector z at the source z(j)=s(i)*X|_ij^12
            # ? for P-SV and z(j)=s(i)*X(5,i) for SH.
            # ?  input:
            # ?       s(3,6)  ---- source coef. for n=0,1,2
            # ?       g(7)    ---- g vector used to construct matrix X|_ij^12
            # ?                    |  0   g1  g2 -g3 |
            # ?        X|_ij^12 =  | -g1  0   g3  g4 | for P-SV.
            # ?                    | -g2 -g3  0   g5 |
            # ?                    |  g3 -g4 -g5  0  |
            # ?        X(5,i) = ( g6 g7 )     for SH.
            # ?  output:
            # ?       z(3,5)  ---- z vector for n=0,1,2
            initialZ(zzz, ggg, sss)
            # * end initialZ(ss, g, z)
        if ilayer < src_layer:
            if ilayer >= rcv_layer:
                # * begin haskellMatrix(a)
                # ? compute 4x4 P-SV Haskell a for the layer
                haskellMatrix(aaa, Ca, Ya, Xa, Cb, Yb,
                              Xb, exa, exb, r, r1, mu2)
                # * end haskellMatrix(a)

                # * begin propagateZ(a, z)
                propagateZ(zzz, aaa, zzz_temp)
                # * end propagateZ(a, z)
            else:
                # * begin propagateB(c, b)

                propagateB(bbb, ccc, bbb_temp)
                # * end propagateB(c, b)
    # c add the top halfspace boundary condition

    cdef:
        double complex rayl = 0. + 0.j, love = 0. + 0.j, val
    for index in range(5):
        rayl = rayl + ggg[index] * eee[index]
    for index in range(5, 7):
        love = love + ggg[index] * eee[index]
    ggg[:4] = 0. + 0.j
    for imat in range(4):
        for jmat in range(5):
            ggg[imat] = ggg[imat] + bbb[imat, jmat] * eee[jmat]
    ggg[2] /= 2.
    ggg[5] = bbb[5, 5] * eee[5] + bbb[5, 6] * eee[6]
    for index in range(3):
        val = zzz[index, 1] * ggg[0] + zzz[index, 2] * \
            ggg[1] - zzz[index, 3] * ggg[2]
        zzz[index, 1] = -zzz[index, 0] * ggg[0] + \
            zzz[index, 2] * ggg[2] + zzz[index, 3] * ggg[3]
        zzz[index, 0] = val
        zzz[index, 4] = zzz[index, 4] * ggg[5]
    val = k
    if src_type == 1:
        val = 1.
    for index in range(3):
        u[index, 0] = u[index, 0] + val * zzz[index, 1] / rayl
        u[index, 1] = u[index, 1] + val * zzz[index, 0] / rayl
        u[index, 2] = u[index, 2] + val * zzz[index, 4] / love


cdef inline(double complex, double complex, double complex, double) sh_ch(double complex a, double kd) nogil:
    # ? compute c=cosh(a*kd); y=sinh(a*kd)/a; x=sinh(a*kd)*a
    # ? and multiply them by ex=exp(-Real(a*kd)) to supress overflow
    cdef:
        double r, i, ex
        double complex c, y, x
    y = kd * a
    r = creal(y)
    i = cimag(y)
    ex = exp(-r)
    y = 0.5 * (cos(i) + sin(i) * 1j)
    x = ex * ex * conj(y)
    c = y + x
    x = y - x
    y = x / a
    x = x * a
    return c, y, x, ex

cdef inline void initialG(double complex[::1] ggg, double complex r, double complex ra, double complex rb, double complex r1, double mu2):
    cdef double complex delta

    delta = r * (1. - ra * rb) - 1.
    ggg[0] = mu2 * (delta - r1)
    ggg[1] = ra
    ggg[2] = delta
    ggg[3] = -rb
    ggg[4] = (1. + delta) / mu2
    # sh, use the 5th row of E^-1, see EQ A4 on ZR/p625, 1/2 omitted
    ggg[5] = -1.
    ggg[6] = 2. / (rb * mu2)

cdef inline void eVector(double complex[::1] eee, double complex ra, double complex rb, double complex r1, double mu2):
    eee[0] = ra * rb - 1.
    eee[1] = mu2 * rb * (1. - r1)
    eee[2] = mu2 * (r1 - ra * rb)
    eee[3] = mu2 * ra * (r1 - 1.)
    eee[4] = mu2 * mu2 * (ra * rb - r1 * r1)
    # c sh part
    eee[5] = -1.
    eee[6] = mu2 * rb / 2.

cdef inline void compoundMatrix(double complex[:, ::1] ccc, double complex Ca, double complex Ya, double complex Xa, double complex Cb,
                                double complex Yb, double complex Xb, double exa, double exb, double complex r, double complex r1,
                                double complex r2, double complex r3, double mu2):
    cdef:
        double complex CaCb, CaYb, CaXb, XaCb, XaXb, YaCb, YaYb
        double ex

    # Ca, Ya, Xa, exa = sh_ch(ra, kd)
    # Cb, Yb, Xb, exb = sh_ch(rb, kd)

    CaCb = Ca * Cb
    CaYb = Ca * Yb
    CaXb = Ca * Xb
    XaCb = Xa * Cb
    XaXb = Xa * Xb
    YaCb = Ya * Cb
    YaYb = Ya * Yb
    ex = exa * exb
    r2 = r * r
    r3 = r1 * r1

    # c p-sv, scaled by exa*exb to supress overflow
    ccc[0, 0] = ((1. + r3) * CaCb - XaXb -
                 r3 * YaYb - 2. * r1 * ex) * r2
    ccc[0, 1] = (XaCb - CaYb) * r / mu2
    ccc[0, 2] = ((1. + r1) * (CaCb - ex) - XaXb - r1 * YaYb) * r2 / mu2
    ccc[0, 3] = (YaCb - CaXb) * r / mu2
    ccc[0, 4] = (2. * (CaCb - ex) - XaXb - YaYb) * r2 / (mu2 * mu2)

    ccc[1, 0] = (r3 * YaCb - CaXb) * r * mu2
    ccc[1, 1] = CaCb
    ccc[1, 2] = (r1 * YaCb - CaXb) * r
    ccc[1, 3] = -Ya * Xb
    ccc[1, 4] = ccc[0, 3]

    ccc[2, 0] = 2. * mu2 * r2 * \
        (r1 * r3 * YaYb - (CaCb - ex) * (r3 + r1) + XaXb)
    ccc[2, 1] = 2. * r * (r1 * CaYb - XaCb)
    ccc[2, 2] = 2. * (CaCb - ccc[0, 0]) + ex
    ccc[2, 3] = -2. * ccc[1, 2]
    ccc[2, 4] = -2. * ccc[0, 2]

    ccc[3, 0] = mu2 * r * (XaCb - r3 * CaYb)
    ccc[3, 1] = -Xa * Yb
    ccc[3, 2] = -ccc[2, 1] / 2.
    ccc[3, 3] = ccc[1, 1]
    ccc[3, 4] = ccc[0, 1]

    ccc[4, 0] = mu2 * mu2 * r2 * \
        (2. * (CaCb - ex) * r3 - XaXb - r3 * r3 * YaYb)
    ccc[4, 1] = ccc[3, 0]
    ccc[4, 2] = -ccc[2, 0] / 2.
    ccc[4, 3] = ccc[1, 0]
    ccc[4, 4] = ccc[0, 0]

    # c sh, scaled by exb
    ccc[5, 5] = Cb
    ccc[5, 6] = -2. * Yb / mu2
    ccc[6, 5] = -mu2 * Xb / 2.
    ccc[6, 6] = Cb

cdef inline void propagateG(double complex[::1] ggg, double complex[:, ::1] ccc, double complex[::1] ggg_temp):
    cdef Py_ssize_t imat, jmat

    ggg_temp[:] = 0. + 0.j
    for imat in range(7):
        for jmat in range(7):
            ggg_temp[imat] = ggg_temp[imat] + \
                ggg[jmat] * ccc[jmat, imat]
    ggg[:] = ggg_temp[:]


cdef inline void separatS(double complex[:, ::1] sss, int updn, int src_type, double complex ra, double complex rb,
                          double complex r, double mu2, double complex[:, ::1] temppp, double complex r1, double[:, :] si):
    cdef:
        int isrc_type
        double complex ra1, rb1, dum, temp_sh
        Py_ssize_t jmat, kmat, index
        int imat
        double complex ctemp

    if updn == 0:
        if src_type == 2:
            for index in range(6):
                for isrc_type in range(3):
                    sss[isrc_type, index] = si[isrc_type, index]
        elif src_type == 1:
            for index in range(6):
                for isrc_type in range(2):
                    sss[isrc_type, index] = si[isrc_type, index]
        else:
            for index in range(6):
                for isrc_type in range(1):
                    sss[isrc_type, index] = si[isrc_type, index]
    else:
        ra1 = 1. / ra
        rb1 = 1. / rb
        if updn == 1:
            dum = -r
            temp_sh = (-1 * 2 / mu2) * rb1
        else:
            dum = r
            temp_sh = (1 * 2 / mu2) * rb1

        temppp[0, 0] = 1.
        temppp[0, 1] = dum * (rb - r1 * ra1)
        temppp[0, 2] = 0
        temppp[0, 3] = dum * (ra1 - rb) / mu2
        temppp[1, 0] = dum * (ra - r1 * rb1)
        temppp[1, 1] = 1.
        temppp[1, 2] = dum * (rb1 - ra) / mu2
        temppp[1, 3] = 0
        temppp[2, 0] = 0
        temppp[2, 1] = dum * (rb - r1 * r1 * ra1) * mu2
        temppp[2, 2] = 1.
        temppp[2, 3] = dum * (r1 * ra1 - rb)
        temppp[3, 0] = dum * (ra - r1 * r1 * rb1) * mu2
        temppp[3, 1] = 0
        temppp[3, 2] = dum * (r1 * rb1 - ra)
        temppp[3, 3] = 1.

        # sss (3,6) si (3,6) tempp (4,4)
        for imat in range(src_type + 1):
            for kmat in range(4):
                ctemp = 0. + 0.j
                for jmat in range(4):
                    ctemp = ctemp + si[imat, jmat] * \
                        temppp.T[jmat, kmat]
                sss[imat, kmat] = ctemp / 2.
            sss[imat, 4] = (si[imat, 4] + temp_sh * si[imat, 5]) / 2.
            sss[imat, 5] = (si[imat, 5] + si[imat, 4] / temp_sh) / 2.

cdef inline void initialZ(double complex[:, ::1] zzz, double complex[::1] ggg, double complex[:, ::1] sss):
    cdef Py_ssize_t index
    for index in range(3):
        # c for p-sv, see WH p1018
        zzz[index, 0] = -sss[index, 1] * ggg[0] - \
            sss[index, 2] * ggg[1] + sss[index, 3] * ggg[2]
        zzz[index, 1] = sss[index, 0] * ggg[0] - \
            sss[index, 2] * ggg[2] - sss[index, 3] * ggg[3]
        zzz[index, 2] = sss[index, 0] * ggg[1] + \
            sss[index, 1] * ggg[2] - sss[index, 3] * ggg[4]
        zzz[index, 3] = -sss[index, 0] * ggg[2] + \
            sss[index, 1] * ggg[3] + sss[index, 2] * ggg[4]
        # c for sh
        zzz[index, 4] = sss[index, 4] * ggg[5] + sss[index, 5] * ggg[6]


cdef inline void haskellMatrix(double complex[:, ::1] aaa, double complex Ca, double complex Ya, double complex Xa, double complex Cb,
                               double complex Yb, double complex Xb, double exa, double exb, double complex r, double complex r1,  double mu2):
    Ca = Ca * exb
    Xa = Xa * exb
    Ya = Ya * exb
    Cb = Cb * exa
    Yb = Yb * exa
    Xb = Xb * exa
    # c p-sv, scaled by exa*exb, see p381/Haskell1964 or EQ 17 of
    # ZR
    aaa[0, 0] = r * (Ca - r1 * Cb)
    aaa[0, 1] = r * (r1 * Ya - Xb)
    aaa[0, 2] = (Cb - Ca) * r / mu2
    aaa[0, 3] = (Xb - Ya) * r / mu2

    aaa[1, 0] = r * (r1 * Yb - Xa)
    aaa[1, 1] = r * (Cb - r1 * Ca)
    aaa[1, 2] = (Xa - Yb) * r / mu2
    aaa[1, 3] = -aaa[0, 2]

    aaa[2, 0] = mu2 * r * r1 * (Ca - Cb)
    aaa[2, 1] = mu2 * r * (r1 * r1 * Ya - Xb)
    aaa[2, 2] = aaa[1, 1]
    aaa[2, 3] = -aaa[0, 1]

    aaa[3, 0] = mu2 * r * (r1 * r1 * Yb - Xa)
    aaa[3, 1] = -aaa[2, 0]
    aaa[3, 2] = -aaa[1, 0]
    aaa[3, 3] = aaa[0, 0]

    # c sh, the Haskell matrix is not needed. it is replaced by exb
    aaa[4, 4] = exb

cdef inline void propagateZ(double complex[:, ::1] zzz, double complex[:, ::1] aaa, double complex[:, ::1] zzz_temp):
    cdef Py_ssize_t imat, jmat, kmat

    zzz_temp[:, :] = 0. + 0.j
    for imat in range(3):
        for jmat in range(5):
            for kmat in range(5):
                zzz_temp[imat, kmat] = zzz_temp[imat,
                                                kmat] + zzz[imat, jmat] * aaa[jmat, kmat]
    zzz[:, :] = zzz_temp[:, :]

cdef inline void propagateB(double complex[:, :] bbb, double complex[:, :] ccc, double complex[:, :] bbb_temp):

    cdef Py_ssize_t imat, jmat, kmat

    bbb_temp[:, :] = 0. + 0.j
    for imat in range(7):
        for jmat in range(7):
            for kmat in range(7):
                bbb_temp[imat, kmat] = bbb_temp[imat,
                                                kmat] + bbb[imat, jmat] * ccc[jmat, kmat]
    bbb[:, :] = bbb_temp[:, :]
