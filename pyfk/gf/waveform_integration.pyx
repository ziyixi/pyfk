#!python
#cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, embedsignature=True, linetrace=True
# note even we have linetrace=True, it still need to be enabled by define_macros=[("CYTHON_TRACE_NOGIL", "1")]
from scipy.special.cython_special cimport jv
import numpy as np
from libc.math cimport fabs, pi, cos, sin
from pyfk.gf.complex cimport clog
from pyfk.setting import SIGMA

cdef tuple calbessel(int nfft2, double dw, double pmin, double dk, double kc, double pmax, double[:] receiver_distance,
                     int wc1):
    cdef:
        double z = pmax*nfft2*dw/kc, k = (z**2+1)**0.5
        double kc2 = kc**2
        int row=nfft2-wc1+1
        int column=int((fabs(kc2+(pmax*(nfft2*dw)))-k)/dk)
        double[:,:,:] aj0list=np.zeros((row,column,len(receiver_distance)))
        double[:,:,:] aj1list=np.zeros((row,column,len(receiver_distance)))
        double[:,:,:] aj2list=np.zeros((row,column,len(receiver_distance)))

    # * main loop for calculating the first order bessel function
    cdef:
        int j,index_n,index_receiver
        double omega
        int n
        double[:,:] klist,xlist,zzlist
    for j in range(row):
        omega = (j+wc1-1)*dw
        k = omega*pmin+0.5*dk
        n = int((fabs(kc2+(pmax*omega))-k)/dk)
        klist = np.zeros(len(receiver_distance), n)
        for index_receiver in range(len(receiver_distance)):
            for index_n in range(n):
                klist[index_receiver,index_n]=k+index_n*dk
        xlist=np.diag(receiver_distance)
        zzlist=np.asarray(klist).T@np.asarray(xlist)
        for index_n in range(n):
            for index_receiver in range(len(receiver_distance)):
                aj0list[j, index_n, index_receiver] = jv(0., zzlist[index_n,index_receiver])
                aj1list[j, index_n, index_receiver] = jv(1., zzlist[index_n,index_receiver])
                aj2list[j, index_n, index_receiver] = jv(2., zzlist[index_n,index_receiver])
    return aj0list,aj1list,aj2list

cdef double complex[:,:] kernel(double k, double complex[:,:] u, double complex[:] kp, double complex[:] ks,
            double complex[:,:] aaa, double complex[:,:] bbb, double complex[:,:] ccc, double complex[:] eee,
            double complex[:] ggg, double complex[:,:] zzz, double complex[:,:] sss, double complex[:,:] temppp,
            double[:] mu, double[:,:] si, double[:] thickness):
    return u


cdef double complex[:,:,:] waveform_integration(int nfft2, double dw, double pmin, double dk, double kc, double pmax,
                           double[:] receiver_distance, int wc1, double smth,
                           double[:] vs, double[:] vp, double[:] qs, double[:] qp,
                           bint flip, double filter_const, bint dynamic, int wc2, double t0, str src_type,
                           double taper, double wc, double[:] mu, double[:] thickness, double[:,:] si):
    # * get jv
    cdef double[:,:,:] aj0list,aj1list,aj2list
    aj0list, aj1list, aj2list=calbessel(nfft2, dw, pmin, dk, kc, pmax, receiver_distance, wc1)

    # * init some values
    cdef double complex[:,:,:] sum_waveform=np.zeros((len(receiver_distance), 9, int(nfft2*smth)), dtype=np.complex)
    cdef double sigma=SIGMA

    # # * main loop, the index in ik, means each wave number
    cdef:
        int ik, idep, n, i, irec, flip_val, icom
        double ztemp, k, kc2, omega, z, aj0, aj1, aj2, filtering, phi
        double complex w, att, atttemp, nf
    # * init some arrays
    cdef:
        double complex[:] kp = np.zeros(len(thickness), dtype=np.complex)
        double complex[:] ks = np.zeros(len(thickness), dtype=np.complex)
        double complex[:,:] u = np.zeros((3, 3), dtype=np.complex)
        double complex[:,:] aaa = np.zeros((5, 5), dtype=np.complex)
        double complex[:,:] bbb = np.zeros((7, 7), dtype=np.complex)
        double complex[:,:] ccc = np.zeros((7, 7), dtype=np.complex)
        double complex[:] eee = np.zeros(7, dtype=np.complex)
        double complex[:] ggg = np.zeros(7, dtype=np.complex)
        double complex[:,:] zzz = np.zeros((3, 5), dtype=np.complex)
        double complex[:,:] sss = np.zeros((3, 6), dtype=np.complex)
        double complex[:,:] temppp = np.zeros((4, 4), dtype=np.complex)
    for ik in range(wc1, nfft2):
        # * the code below is modified from FK
        ztemp = pmax*nfft2*dw/kc
        kc2=kc**2
        omega=ik*dw
        w = omega-sigma*1j
        # apply attenuation
        att = clog(w/(2*pi))/pi+0.5j
        for idep in range(len(thickness)):
            kp[idep]= (w/(vp[idep]*(1.+att/qp[idep])))**2
            ks[idep] = (w/(vs[idep]*(1.+att/qs[idep])))**2
        k = omega*pmin+0.5*dk
        n = int(((kc2+(pmax*omega)**2)**0.5-k)/dk)
        if flip:
            flip_val = -1
        else:
            flip_val = 1
        for i in range(n):
            u = kernel(k, u, kp, ks, aaa, bbb, ccc, eee, ggg, zzz, sss, temppp, mu, si, thickness)
            # * loop irec to get the value of sum_waveform
            for irec in range(len(receiver_distance)):
                aj0 = aj0list[ik-wc1+1, i, irec]
                aj1 = aj1list[ik-wc1+1, i, irec]
                aj2 = aj2list[ik-wc1+1, i, irec]
                z = k*receiver_distance[irec]
                # do the numerical integration here
                sum_waveform[irec, 0, ik] += u[0, 0]*aj0*flip_val
                sum_waveform[irec, 1, ik] += -u[0, 1]*aj1
                sum_waveform[irec, 2, ik] += -u[0, 2]*aj1

                nf = (u[1, 1]+u[1, 2])*aj1/z
                sum_waveform[irec, 3, ik] += u[1, 0]*aj1*flip_val
                sum_waveform[irec, 4, ik] += u[1, 1]*aj0-nf
                sum_waveform[irec, 5, ik] += u[1, 2]*aj0-nf

                nf = 2.*(u[2, 1]+u[2, 2])*aj2/z
                sum_waveform[irec, 6, ik] += u[2, 0]*aj2*flip_val
                sum_waveform[irec, 7, ik] += u[2, 1]*aj1-nf
                sum_waveform[irec, 8, ik] += u[2, 2]*aj1-nf

                k = k+dk
        # * for each ik, we apply the filtering and apply the time shift in the frequency domain
        filtering = filter_const
        if dynamic and (ik+1 > wc):
            filtering = 0.5*(1.+cos((ik+1-wc)*taper))*filtering
        if dynamic and (ik+1 < wc2):
            filtering = 0.5*(1.+cos((wc2-ik-1)*pi/(wc2-wc1)))*filtering
        phi = omega*t0
        atttemp = filtering*(cos(phi)+sin(phi)*1j)
        # in fk's code, only apply atttemp for ncom, here we apply to all, with no difference
        for icom in range(9):
            for irec in range(len(receiver_distance)):
                sum_waveform[irec, icom, ik] *= atttemp
    return sum_waveform
