# write cuda kernels with numba, parallel for each ibool

import numpy as np
from numba import complex128, cuda, njit
from pyfk.gf.cuda.utils import (cal_cujn, compoundMatrix, eVector,
                                haskellMatrix, initialG, initialZ, propagateB,
                                propagateG, propagateZ, separatS, sh_ch)


def _waveform_integration(
        nfft2: int,
        dw: float,
        pmin: float,
        dk: float,
        kc: float,
        pmax: float,
        receiver_distance: np.ndarray,
        wc1: int,
        vs: np.ndarray,
        vp: np.ndarray,
        qs: np.ndarray,
        qp: np.ndarray,
        flip: bool,
        filter_const: float,
        dynamic: bool,
        wc2: int,
        t0: int,
        src_type: int,
        taper: float,
        wc: int,
        mu: np.ndarray,
        thickness: np.ndarray,
        si: np.ndarray,
        src_layer: int,
        rcv_layer: int,
        updn: int,
        epsilon: float,
        sigma: float,
        sum_waveform: np.ndarray,
        cuda_divide_num: int):
    # * generate kp and ks array ((nfft2-wc1+1)*len(thickness))
    kp_list = np.zeros((nfft2-wc1+1, len(thickness)), dtype=complex)
    ks_list = np.zeros((nfft2-wc1+1, len(thickness)), dtype=complex)

    # * get the n list, kp and ks
    n_list: np.ndarray = np.zeros(nfft2-wc1+1, dtype=np.int)
    n_list_accumulate: np.ndarray = np.zeros(nfft2-wc1+1, dtype=np.int)
    get_n_list_kpks(wc1, nfft2, kc, dw, pmin, pmax, dk, sigma, thickness, vp, vs, qp, qs,
                    n_list, n_list_accumulate, kp_list, ks_list)
    n_all: int = n_list_accumulate[-1]

    for index_cuda_divide in range(cuda_divide_num):
        # * current index_cuda_divide info
        current_range_list = np.array_split(range(n_all), cuda_divide_num)[
            index_cuda_divide]
        current_n_all = len(current_range_list)
        current_offset = current_range_list[0]

        # * generate the ik and i list representing the i wavenumber and i frequency
        ik_list = np.zeros(current_n_all, dtype=np.int)
        i_list = np.zeros(current_n_all, dtype=np.int)

        # * call fill_vals cuda kernel
        threadsperblock = 128
        blockspergrid = (current_n_all + (threadsperblock - 1)
                         ) // threadsperblock

        ik_list_d = cuda.to_device(ik_list)
        i_list_d = cuda.to_device(i_list)
        fill_vals[blockspergrid, threadsperblock](
            n_list, n_list_accumulate, ik_list_d, i_list_d, wc1, nfft2, current_n_all, current_offset)
        ik_list = ik_list_d.copy_to_host()
        i_list = i_list_d.copy_to_host()

        # * initialize the big matrix u (current_n_all*3*3)
        u: np.ndarray = np.zeros((current_n_all, 3, 3), dtype=complex)

        # * init cuda arrays
        # u, ik_list, i_list, kp, ks, thickness, mu, si
        u_d = cuda.to_device(u)
        ik_list_d = cuda.to_device(ik_list)
        i_list_d = cuda.to_device(i_list)
        kp_list_d = cuda.to_device(kp_list)
        ks_list_d = cuda.to_device(ks_list)
        thickness_d = cuda.to_device(thickness)
        mu_d = cuda.to_device(mu)
        si_d = cuda.to_device(si)

        # * run the cuda kernel function
        parallel_kernel[blockspergrid, threadsperblock](u_d, ik_list_d, i_list_d, kp_list_d, ks_list_d, thickness_d, mu_d, si_d,
                                                        dw, pmin, dk, src_layer, rcv_layer, updn, src_type, epsilon, wc1, current_n_all)
        u = u_d.copy_to_host()

        # * get sum_waveform
        flip_val = 0
        if flip:
            flip_val = -1.
        else:
            flip_val = 1.

        z_list = np.zeros(
            (current_n_all, len(receiver_distance)), dtype=float)
        get_z_list(z_list, ik_list, i_list, receiver_distance,
                   dw, pmin, dk, current_n_all)
        aj0_list = cal_cujn(0, z_list)
        aj1_list = cal_cujn(1, z_list)
        aj2_list = cal_cujn(2, z_list)
        # it's not appropriate to use cuda here, as it will use large atomic operation.
        get_sum_waveform(sum_waveform, u, ik_list, receiver_distance,  flip_val,
                         z_list, aj0_list, aj1_list, aj2_list, current_n_all)

    # * perform the filtering
    apply_filter(wc1, nfft2, dw, filter_const, dynamic, wc, taper, wc2,
                 receiver_distance, t0, sum_waveform)


@njit("void(int64,int64,float64,float64,float64,float64,float64,float64,float64[:],float64[:],float64[:],float64[:],float64[:],int64[:],int64[:],complex128[:,:],complex128[:,:])")
def get_n_list_kpks(wc1: int, nfft2: int, kc: float, dw: float, pmin: float, pmax: float, dk: float, sigma: float,
                    thickness: np.ndarray, vp: np.ndarray, vs: np.ndarray, qp: np.ndarray, qs: np.ndarray,
                    # output
                    n_list: np.ndarray, n_list_accumulate: np.ndarray,  kp_list: np.ndarray, ks_list: np.ndarray):
    n_all = 0
    for ik in range(wc1 - 1, nfft2):
        kc2 = kc**2
        omega = ik * dw
        w = omega - sigma * 1j
        att = np.log(w / (2 * np.pi)) / np.pi + 0.5j
        k = omega * pmin + 0.5 * dk
        n = int(((kc2 + (pmax * omega)**2)**0.5 - k) / dk)
        n_list[ik-wc1+1] = n
        n_all = n_all+n
        n_list_accumulate[ik-wc1+1] = n_all
        for idep in range(len(thickness)):
            kp_list[ik-wc1+1, idep] = (w /
                                       (vp[idep] * (1. + att / qp[idep])))**2
            ks_list[ik-wc1+1, idep] = (w /
                                       (vs[idep] * (1. + att / qs[idep])))**2


@cuda.jit("void(int64[:],int64[:],int64[:],int64[:],int64,int64,int64,int64)")
def fill_vals(n_list: np.ndarray, n_list_accumulate: np.ndarray, ik_list: np.ndarray, i_list: np.ndarray, wc1: int, nfft2: int, current_n_all: int, current_offset: int):
    pos = cuda.grid(1)
    if pos < current_n_all:
        ibool = pos+current_offset
        for isearch in range(nfft2-wc1+1):
            if(n_list_accumulate[isearch] > ibool):
                ik_list[pos] = isearch+wc1 - 1
                i_list[pos] = ibool - \
                    (n_list_accumulate[isearch]-n_list[isearch])
                break


@njit("void(float64[:,:],int64[:],int64[:],float64[:],float64,float64,float64,int64)")
def get_z_list(z_list, ik_list, i_list, receiver_distance, dw, pmin, dk, current_n_all):
    for pos in range(current_n_all):
        ik = ik_list[pos]
        i = i_list[pos]
        omega = ik * dw
        k = omega * pmin + (0.5+i) * dk
        for irec in range(len(receiver_distance)):
            z_list[pos, irec] = k * receiver_distance[irec]


@njit("void(complex128[:,:,:],complex128[:,:,:],int64[:],float64[:],float64,float64[:,:],float64[:,:],float64[:,:],float64[:,:],int64)")
def get_sum_waveform(sum_waveform, u, ik_list, receiver_distance,  flip_val,
                     z_list, aj0_list, aj1_list, aj2_list, current_n_all):
    for pos in range(current_n_all):
        ik = ik_list[pos]
        for irec in range(len(receiver_distance)):
            z = z_list[pos, irec]
            aj0 = aj0_list[pos, irec]
            aj1 = aj1_list[pos, irec]
            aj2 = aj2_list[pos, irec]
            # do the numerical integration here
            sum_waveform[irec, 0, ik] += u[pos, 0, 0] * \
                aj0 * flip_val
            sum_waveform[irec, 1, ik] += -u[pos, 0, 1] * aj1
            sum_waveform[irec, 2, ik] += -u[pos, 0, 2] * aj1

            nf = (u[pos, 1, 1] + u[pos, 1, 2]) * aj1 / z
            sum_waveform[irec, 3, ik] += u[pos, 1, 0] * \
                aj1 * flip_val
            sum_waveform[irec, 4, ik] += u[pos, 1, 1] * aj0 - nf
            sum_waveform[irec, 5, ik] += u[pos, 1, 2] * aj0 - nf

            nf = 2. * (u[pos, 2, 1] + u[pos, 2, 2]) * aj2 / z
            sum_waveform[irec, 6, ik] += u[pos, 2, 0] * \
                aj2 * flip_val
            sum_waveform[irec, 7, ik] += u[pos, 2, 1] * aj1 - nf
            sum_waveform[irec, 8, ik] += u[pos, 2, 2] * aj1 - nf


@njit("void(int64,int64,float64,float64,boolean,int64,float64,int64,float64[:],float64[:],complex128[:,:,:])")
def apply_filter(wc1, nfft2, dw, filter_const, dynamic, wc, taper, wc2,
                 receiver_distance, t0, sum_waveform):
    for ik in range(wc1 - 1, nfft2):
        omega = ik * dw
        # * for each ik, we apply the filtering and apply the time shift in the frequency domain
        filtering = filter_const
        if dynamic and (ik + 1 > wc):
            filtering = 0.5 * (1. + np.cos((ik + 1 - wc) * taper)) * filtering
        if dynamic and (ik + 1 < wc2):
            filtering = 0.5 * \
                (1. + np.cos((wc2 - ik - 1) * np.pi / (wc2 - wc1))) * filtering
        # in fk's code, only apply atttemp for ncom, here we apply to all, with
        # no difference
        for icom in range(9):
            for irec in range(len(receiver_distance)):
                phi = omega * t0[irec]
                atttemp = filtering * (np.cos(phi) + np.sin(phi) * 1j)
                sum_waveform[irec, icom,
                             ik] = sum_waveform[irec, icom, ik]*atttemp


@cuda.jit("void(complex128[:,:,:],int64[:],int64[:],complex128[:,:],complex128[:,:],float64[:],float64[:],float64[:,:],float64,float64,float64,int64,int64,int64,int64,float64,int64,int64)")
def parallel_kernel(u, ik_list, i_list, kp_list, ks_list, thickness, mu, si,
                    dw, pmin, dk, src_layer, rcv_layer, updn, src_type, epsilon, wc1, current_n_all):
    # * pos for this thread
    pos = cuda.grid(1)
    if(pos < current_n_all):
        ik = ik_list[pos]
        i = i_list[pos]

        # * generate some local arrays
        aaa = cuda.local.array((5, 5), dtype=complex128)
        bbb = cuda.local.array((7, 7), dtype=complex128)
        ccc = cuda.local.array((7, 7), dtype=complex128)
        eee = cuda.local.array(7, dtype=complex128)
        ggg = cuda.local.array(7, dtype=complex128)
        zzz = cuda.local.array((3, 5), dtype=complex128)
        sss = cuda.local.array((3, 6), dtype=complex128)
        temppp = cuda.local.array((4, 4), dtype=complex128)

        omega = ik * dw
        k = omega * pmin + (0.5+i) * dk

        # * here we directly run the kernel function as in cython
        aaa[:, :] = 0. + 0.j
        bbb[:, :] = 0. + 0.j
        ccc[:, :] = 0. + 0.j
        eee[:] = 0. + 0.j
        ggg[:] = 0. + 0.j
        zzz[:, :] = 0. + 0.j
        sss[:, :] = 0. + 0.j
        temppp[:, :] = 0. + 0.j

        for index in range(7):
            bbb[index, index] = 1.
        eee[0], eee[5], ggg[4], ggg[6] = 1., 1., 1., 1.

        for ilayer in range(len(thickness) - 1, -1, -1):
            Ca, Ya, Xa, Cb, Yb, Xb = 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j
            exa, exb = 0., 0.
            kka = kp_list[ik-wc1+1, ilayer] / (k**2)
            kkb = ks_list[ik-wc1+1, ilayer] / (k**2)
            r = 2. / kkb
            kd = k * thickness[ilayer]
            mu2 = 2. * mu[ilayer]
            ra = (1. - kka)**0.5
            rb = (1. - kkb)**0.5
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
                               exa, exb, r, r1,  mu2)
                # * end compoundMatrix(c)

                # * begin propagateG(c, g)
                # ? propagate g vector upward using the compound matrix
                # ?       g = g*a
                propagateG(ggg, ccc)
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
                    propagateZ(zzz, aaa)
                    # * end propagateZ(a, z)
                else:
                    # * begin propagateB(c, b)

                    propagateB(bbb, ccc)
                    # * end propagateB(c, b)
        # c add the top halfspace boundary condition

        rayl, love = 0. + 0.j, 0. + 0.j
        for index in range(5):
            rayl = rayl + ggg[index] * eee[index]
        for index in range(5, 7):
            love = love + ggg[index] * eee[index]
        for index in range(4):
            ggg[index] = 0. + 0.j
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

        # * set u
        for index in range(3):
            u[pos, index, 0] = u[pos, index, 0] + val * zzz[index, 1] / rayl
            u[pos, index, 1] = u[pos, index, 1] + val * zzz[index, 0] / rayl
            u[pos, index, 2] = u[pos, index, 2] + val * zzz[index, 4] / love
