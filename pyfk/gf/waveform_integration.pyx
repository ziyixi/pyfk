#!python
# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, linetrace=True
# note even we have linetrace=True, it still need to be enabled by
# define_macros=[("CYTHON_TRACE_NOGIL", "1")]
import numpy as np

from pyfk.config.config import SeisModel, Config
from pyfk.setting import EPSILON

from scipy.special.cython_special cimport jv
from libc.math cimport fabs, pi, cos, sin, exp
from pyfk.utils.complex cimport clog, csqrt, creal, cimag, conj

cdef tuple calbessel(int nfft2, double dw, double pmin, double dk, double kc, double pmax, double[:] receiver_distance,
                     int wc1):
    cdef:
        double z = pmax * nfft2 * dw / kc, k = (z**2 + 1)**0.5
        double kc2 = kc**2
        int row = nfft2 - wc1 + 1
        int column = 0

    # * find the appropriate value of column
    cdef:
        int j, index_n, index_receiver
        double omega
        int n
        double[:, :] klist, xlist, zzlist
    for j in range(row):
        omega = (j + wc1 - 1) * dw
        k = omega * pmin + 0.5 * dk
        n = int(((kc2 + (pmax * omega)**2)**0.5 - k) / dk)
        if column < n:
            column = n

    cdef:
        double[:, :, :] aj0list = np.zeros((row, column, len(receiver_distance)))
        double[:, :, :] aj1list = np.zeros((row, column, len(receiver_distance)))
        double[:, :, :] aj2list = np.zeros((row, column, len(receiver_distance)))

    # * main loop for calculating the first order bessel function
    for j in range(row):
        omega = (j + wc1 - 1) * dw
        k = omega * pmin + 0.5 * dk
        n = int(((kc2 + (pmax * omega)**2)**0.5 - k) / dk)
        klist = np.zeros((len(receiver_distance), n), dtype=np.float64)
        for index_receiver in range(len(receiver_distance)):
            for index_n in range(n):
                klist[index_receiver, index_n] = k + index_n * dk
        xlist = np.diag(receiver_distance)
        zzlist = np.asarray(klist).T @ np.asarray(xlist)
        for index_n in range(n):
            for index_receiver in range(len(receiver_distance)):
                aj0list[j, index_n, index_receiver] = jv(
                    0., zzlist[index_n, index_receiver])
                aj1list[j, index_n, index_receiver] = jv(
                    1., zzlist[index_n, index_receiver])
                aj2list[j, index_n, index_receiver] = jv(
                    2., zzlist[index_n, index_receiver])
    return aj0list, aj1list, aj2list


cdef double complex[:, :, :] _waveform_integration(int nfft2, double dw, double pmin, double dk, double kc, double pmax,
                                                   double[:] receiver_distance, int wc1,
                                                   double[:] vs, double[:] vp, double[:] qs, double[:] qp,
                                                   bint flip, double filter_const, bint dynamic, int wc2, double[:] t0, str src_type,
                                                   double taper, int wc, double[:] mu, double[:] thickness, double[:, :] si,
                                                   int src_layer, int rcv_layer, str updn, double epsilon, double sigma):
    # * get jv
    cdef double[:, :, :] aj0list, aj1list, aj2list
    aj0list, aj1list, aj2list = calbessel(
        nfft2, dw, pmin, dk, kc, pmax, receiver_distance, wc1)

    # * init some values
    cdef double complex[:, :, :] sum_waveform = np.zeros((len(receiver_distance), 9, int(nfft2)), dtype=np.complex)

    # # * main loop, the index in ik, means each wave number
    cdef:
        int ik, idep, n, i, irec, flip_val, icom
        double ztemp, k, kc2, omega, z, aj0, aj1, aj2, filtering, phi
        double complex w, att, atttemp, nf
    # * init some arrays
    cdef:
        double complex[:] kp = np.zeros(len(thickness), dtype=np.complex)
        double complex[:] ks = np.zeros(len(thickness), dtype=np.complex)
        double complex[:, :] u = np.zeros((3, 3), dtype=np.complex)
        double complex[:, :] aaa = np.zeros((5, 5), dtype=np.complex)
        double complex[:, :] bbb = np.zeros((7, 7), dtype=np.complex)
        double complex[:, :] ccc = np.zeros((7, 7), dtype=np.complex)
        double complex[:] eee = np.zeros(7, dtype=np.complex)
        double complex[:] ggg = np.zeros(7, dtype=np.complex)
        double complex[:, :] zzz = np.zeros((3, 5), dtype=np.complex)
        double complex[:, :] sss = np.zeros((3, 6), dtype=np.complex)
        double complex[:, :] temppp = np.zeros((4, 4), dtype=np.complex)
    for ik in range(wc1 - 1, nfft2):
        # * the code below is modified from FK
        ztemp = pmax * nfft2 * dw / kc
        kc2 = kc**2
        omega = ik * dw
        w = omega - sigma * 1j
        # apply attenuation
        att = clog(w / (2 * pi)) / pi + 0.5j
        for idep in range(len(thickness)):
            kp[idep] = (w / (vp[idep] * (1. + att / qp[idep])))**2
            ks[idep] = (w / (vs[idep] * (1. + att / qs[idep])))**2
        k = omega * pmin + 0.5 * dk
        n = int(((kc2 + (pmax * omega)**2)**0.5 - k) / dk)
        if flip:
            flip_val = -1
        else:
            flip_val = 1
        for i in range(n):
            u = kernel(
                k,
                u,
                kp,
                ks,
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
                epsilon)
            # * loop irec to get the value of sum_waveform
            for irec in range(len(receiver_distance)):
                aj0 = aj0list[ik - wc1 + 1, i, irec]
                aj1 = aj1list[ik - wc1 + 1, i, irec]
                aj2 = aj2list[ik - wc1 + 1, i, irec]
                z = k * receiver_distance[irec]
                # do the numerical integration here
                sum_waveform[irec, 0, ik] += u[0, 0] * aj0 * flip_val
                sum_waveform[irec, 1, ik] += -u[0, 1] * aj1
                sum_waveform[irec, 2, ik] += -u[0, 2] * aj1

                nf = (u[1, 1] + u[1, 2]) * aj1 / z
                sum_waveform[irec, 3, ik] += u[1, 0] * aj1 * flip_val
                sum_waveform[irec, 4, ik] += u[1, 1] * aj0 - nf
                sum_waveform[irec, 5, ik] += u[1, 2] * aj0 - nf

                nf = 2. * (u[2, 1] + u[2, 2]) * aj2 / z
                sum_waveform[irec, 6, ik] += u[2, 0] * aj2 * flip_val
                sum_waveform[irec, 7, ik] += u[2, 1] * aj1 - nf
                sum_waveform[irec, 8, ik] += u[2, 2] * aj1 - nf

            k = k + dk
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
                sum_waveform[irec, icom, ik] *= atttemp
    return sum_waveform

cdef double complex[:, :] kernel(double k, double complex[:, :] u, double complex[:] kp, double complex[:] ks,
                                 double complex[:, :] aaa, double complex[:, :] bbb, double complex[:, :] ccc, double complex[:] eee,
                                 double complex[:] ggg, double complex[:, :] zzz, double complex[:, :] sss, double complex[:, :] temppp,
                                 double[:] mu, double[:, :] si, double[:] thickness, str src_type, int src_layer, int rcv_layer,
                                 str updn, double epsilon):
    # * the code below is directly modified from the FK code
    cdef int index, ilayer, isrc_type, nsrc_type

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
    u[:, :] = 0.
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
        double complex kka, kkb, r, ra, rb, r1, delta
        # else
        double complex Ca, Ya, Xa, Cb, Yb, Xb
        double complex CaCb, CaYb, CaXb, XaCb, XaXb, YaCb, YaYb
        double exa, exb, ex
        double complex r2, r3
        # ilayer==src_layer
        double complex ra1, rb1, dum, temp_sh
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
            delta = r * (1. - ra * rb) - 1.
            ggg[0] = mu2 * (delta - r1)
            ggg[1] = ra
            ggg[2] = delta
            ggg[3] = -rb
            ggg[4] = (1. + delta) / mu2
            # sh, use the 5th row of E^-1, see EQ A4 on ZR/p625, 1/2 omitted
            ggg[5] = -1.
            ggg[6] = 2. / (rb * mu2)
            # * end initailG
        elif ilayer == 0 and thickness[ilayer] < epsilon:
            # * begin eVector(e)
            # ? The first 5 members are E|_12^ij, ij=12,13,23,24,34.
            # ? The last two are the first column of SH E matrix.
            # For p-sv, compute E|_(12)^(ij), ij=12, 13, 23, 24, 34.
            eee[0] = ra * rb - 1.
            eee[1] = mu2 * rb * (1. - r1)
            eee[2] = mu2 * (r1 - ra * rb)
            eee[3] = mu2 * ra * (r1 - 1.)
            eee[4] = mu2 * mu2 * (ra * rb - r1 * r1)
            # c sh part
            eee[5] = -1.
            eee[6] = mu2 * rb / 2.
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
            Ca, Ya, Xa, exa = sh_ch(Ca, Ya, Xa, exa, ra, kd)
            Cb, Yb, Xb, exb = sh_ch(Cb, Yb, Xb, exb, rb, kd)

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
            # * end compoundMatrix(c)

            # * begin propagateG(c, g)
            # ? propagate g vector upward using the compound matrix
            # ?       g = g*a
            ggg = np.asarray(ggg) @ np.asarray(ccc)
            # * end propagateG(c, g)
        if ilayer == src_layer:
            # * begin separatS
            if updn == "all":
                if src_type == "dc":
                    for index in range(6):
                        for isrc_type in range(3):
                            sss[isrc_type, index] = si[isrc_type, index]
                            nsrc_type = 2
                elif src_type == "sf":
                    for index in range(6):
                        for isrc_type in range(2):
                            sss[isrc_type, index] = si[isrc_type, index]
                            nsrc_type = 1
                else:
                    for index in range(6):
                        for isrc_type in range(1):
                            sss[isrc_type, index] = si[isrc_type, index]
                            nsrc_type = 0
            else:
                ra1 = 1. / ra
                rb1 = 1. / rb
                if updn == "up":
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

                sss[:nsrc_type + 1,
                    :4] = (np.asarray(si)[:nsrc_type + 1,
                                          :4] @ np.asarray(temppp).T) / 2.
                sss[:nsrc_type + 1,
                    4] = (np.asarray(si)[:nsrc_type + 1,
                                         4] + temp_sh * np.asarray(si)[:nsrc_type + 1,
                                                                       5]) / 2.
                sss[:nsrc_type + 1,
                    5] = (np.asarray(si)[:nsrc_type + 1,
                                         5] + np.asarray(si)[:nsrc_type + 1,
                                                             4] / temp_sh) / 2
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
            # * end initialZ(ss, g, z)
        if ilayer < src_layer:
            if ilayer >= rcv_layer:
                # * begin haskellMatrix(a)
                # ? compute 4x4 P-SV Haskell a for the layer
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
                # * end haskellMatrix(a)

                # * begin propagateZ(a, z)
                zzz = np.asarray(zzz) @ np.asarray(aaa)
                # * end propagateZ(a, z)
            else:
                # * begin propagateB(c, b)
                bbb = np.asarray(bbb) @ np.asarray(ccc)
                # * end propagateB(c, b)
    # c add the top halfspace boundary condition
    eee_np = np.asarray(eee)
    ggg_np = np.asarray(ggg)
    bbb_np = np.asarray(bbb)
    zzz_np = np.asarray(zzz)
    rayl = np.sum(ggg_np[:5] * eee_np[:5])
    love = np.sum(ggg_np[5:] * eee_np[5:])
    ggg_np[:4] = (bbb_np[:4, :5] @ eee_np[:5])
    ggg_np[2] /= 2.
    ggg_np[5] = bbb_np[5, 5] * eee_np[5] + bbb_np[5, 6] * eee_np[6]
    for index in range(3):
        val = zzz_np[index, 1] * ggg_np[0] + zzz_np[index, 2] * \
            ggg_np[1] - zzz_np[index, 3] * ggg_np[2]
        zzz_np[index, 1] = -zzz_np[index, 0] * ggg_np[0] + \
            zzz_np[index, 2] * ggg_np[2] + zzz_np[index, 3] * ggg_np[3]
        zzz_np[index, 0] = val
        zzz_np[index, 4] = zzz_np[index, 4] * ggg_np[5]
    val = k
    if src_type == "sf":
        val = 1
    u_np = np.asarray(u)
    u_np[:, 0] = val * zzz_np[:, 1] / rayl
    u_np[:, 1] = val * zzz_np[:, 0] / rayl
    u_np[:, 2] = val * zzz_np[:, 4] / love
    u = u_np

    return u


cdef inline(double complex, double complex, double complex, double) sh_ch(double complex c, double complex y, double complex x,
                                                                          double ex, double complex a, double kd) nogil:
    # ? compute c=cosh(a*kd); y=sinh(a*kd)/a; x=sinh(a*kd)*a
    # ? and multiply them by ex=exp(-Real(a*kd)) to supress overflow
    cdef:
        double r, i
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


def waveform_integration(
        model: SeisModel,
        config: Config,
        src_layer: int,
        rcv_layer: int,
        taper: float,
        pmin: float,
        pmax: float,
        dk: float,
        nfft2: int,
        dw: float,
        kc: float,
        flip: bool,
        filter_const: float,
        dynamic: bool,
        wc1: int,
        wc2: int,
        t0: np.ndarray,
        wc: int,
        si: np.ndarray,
        sigma: float):
    # * note, here we use parameters within config as their value is different from config
    mu = model.rh * model.vs * model.vs
    sum_waveform = np.asarray(
        _waveform_integration(
            nfft2,
            dw,
            pmin,
            dk,
            kc,
            pmax,
            config.receiver_distance,
            wc1,
            model.vs,
            model.vp,
            model.qs,
            model.qp,
            flip,
            filter_const,
            dynamic,
            wc2,
            t0,
            config.source.srcType,
            taper,
            wc,
            mu,
            model.th,
            si,
            src_layer,
            rcv_layer,
            config.updn,
            EPSILON,
            sigma))
    return sum_waveform
