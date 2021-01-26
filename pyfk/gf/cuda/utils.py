# define utils functions used by the main computation
import math

import cupy as cp
import numpy as np
from numba import complex128, cuda


@cuda.jit("Tuple((complex128,complex128,complex128,float64))(complex128,float64)", device=True)
def sh_ch(a, kd):
    y = kd * a
    r = y.real
    i = y.imag
    ex = math.exp(-r)
    y = 0.5 * (math.cos(i) + math.sin(i) * 1j)
    x = ex * ex * y.conjugate()
    c = y + x
    x = y - x
    y = x / a
    x = x * a
    return c, y, x, ex


@cuda.jit("void(complex128[:],complex128,complex128,complex128,complex128,float64)", device=True)
def initialG(ggg, r, ra, rb, r1, mu2):
    delta = r * (1. - ra * rb) - 1.
    ggg[0] = mu2 * (delta - r1)
    ggg[1] = ra
    ggg[2] = delta
    ggg[3] = -rb
    ggg[4] = (1. + delta) / mu2
    # sh, use the 5th row of E^-1, see EQ A4 on ZR/p625, 1/2 omitted
    ggg[5] = -1.
    ggg[6] = 2. / (rb * mu2)


@cuda.jit("void(complex128[:],complex128,complex128,complex128,float64)", device=True)
def eVector(eee, ra, rb, r1, mu2):
    eee[0] = ra * rb - 1.
    eee[1] = mu2 * rb * (1. - r1)
    eee[2] = mu2 * (r1 - ra * rb)
    eee[3] = mu2 * ra * (r1 - 1.)
    eee[4] = mu2 * mu2 * (ra * rb - r1 * r1)
    # c sh part
    eee[5] = -1.
    eee[6] = mu2 * rb / 2.


@cuda.jit("void(complex128[:,:],complex128,complex128,complex128,complex128,complex128,complex128,float64,float64,complex128,complex128,float64)", device=True)
def compoundMatrix(ccc, Ca, Ya, Xa, Cb, Yb, Xb, exa, exb, r, r1, mu2):  # no r2, r3
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


@cuda.jit("void(complex128[:],complex128[:,:])", device=True)
def propagateG(ggg, ccc):
    ggg_temp = cuda.local.array(7, dtype=complex128)
    ggg_temp[:] = 0. + 0.j
    for imat in range(7):
        for jmat in range(7):
            ggg_temp[imat] = ggg_temp[imat] + \
                ggg[jmat] * ccc[jmat, imat]

    for imat in range(7):
        ggg[imat] = ggg_temp[imat]


@cuda.jit("void(complex128[:,:],int64,int64,complex128,complex128,complex128,float64,complex128[:,:],complex128,float64[:,:])", device=True)
def separatS(sss, updn, src_type, ra, rb, r, mu2, temppp, r1, si):
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
                        temppp[kmat, jmat]
                sss[imat, kmat] = ctemp / 2.
            sss[imat, 4] = (si[imat, 4] + temp_sh * si[imat, 5]) / 2.
            sss[imat, 5] = (si[imat, 5] + si[imat, 4] / temp_sh) / 2.


@cuda.jit("void(complex128[:,:],complex128[:],complex128[:,:])", device=True)
def initialZ(zzz, ggg, sss):
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


@cuda.jit("void(complex128[:,:],complex128,complex128,complex128,complex128,complex128,complex128,float64,float64,complex128,complex128,float64)", device=True)
def haskellMatrix(aaa, Ca, Ya, Xa, Cb, Yb, Xb, exa, exb, r, r1, mu2):
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


@cuda.jit("void(complex128[:,:],complex128[:,:])", device=True)
def propagateZ(zzz, aaa):
    zzz_temp = cuda.local.array((3, 5), dtype=complex128)
    zzz_temp[:, :] = 0. + 0.j
    for imat in range(3):
        for jmat in range(5):
            for kmat in range(5):
                zzz_temp[imat, kmat] = zzz_temp[imat,
                                                kmat] + zzz[imat, jmat] * aaa[jmat, kmat]

    for imat in range(3):
        for kmat in range(5):
            zzz[imat, kmat] = zzz_temp[imat, kmat]


@cuda.jit("void(complex128[:,:],complex128[:,:])", device=True)
def propagateB(bbb, ccc):
    bbb_temp = cuda.local.array((7, 7), dtype=complex128)

    bbb_temp[:, :] = 0. + 0.j
    for imat in range(7):
        for jmat in range(7):
            for kmat in range(7):
                bbb_temp[imat, kmat] = bbb_temp[imat,
                                                kmat] + bbb[imat, jmat] * ccc[jmat, kmat]

    for imat in range(7):
        for kmat in range(7):
            bbb[imat, kmat] = bbb_temp[imat, kmat]


cujn = cp.ElementwiseKernel(
    'int64 n, float64 x',
    'float64 z',
    'z=jn(n,x)',
    'cujn')


def cal_cujn(n: int, x: np.ndarray):
    x_cu = cp.asarray(x)
    result_cu = cujn(n, x_cu)
    return cp.asnumpy(result_cu)
