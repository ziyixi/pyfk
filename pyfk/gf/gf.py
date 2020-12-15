from copy import copy
from typing import Optional, Union, List

import numpy as np
from numpy.fft import irfft
from obspy.core.stream import Stream
from obspy.core.trace import Trace
from pyfk.gf.waveform_integration import _waveform_integration
from pyfk.taup.taup import taup

from pyfk.config.config import Config, SeisModel
from pyfk.setting import EPSILON, SIGMA


def calculate_gf(config: Optional[Config] = None) -> Union[List[Stream], List[Stream]]:
    """
    Compute displacements in cm in the up, radial (outward), and transverse (clockwise) directions produced by different seismic sources

    :param config: the configuration of calculating the Green's function, defaults to None
    :type config: Optional[Config], optional
    :return: if npt==2 or 1, return a 2D list, and each row represents the static displacements; otherwise, return a list of Stream, each stream keeps the order of GF as in FK, and the order of streams is the same as the receiver_distance.
    :rtype: Union[List[Stream], List[Stream]]
    """
    # * firstly, we calculate the travel time and ray parameter for vp and vs
    t0_vp: np.ndarray
    td_vp: np.ndarray
    p0_vp: np.ndarray
    pd_vp: np.ndarray
    t0_vs: np.ndarray
    td_vs: np.ndarray
    p0_vs: np.ndarray
    pd_vs: np.ndarray
    t0_vp, td_vp, p0_vp, pd_vp = taup(
        config.src_layer, config.rcv_layer, config.model.th.astype(
            np.float64), config.model.vp.astype(
            np.float64), config.receiver_distance.astype(
                np.float64))
    t0_vs, td_vs, p0_vs, pd_vs = taup(
        config.src_layer, config.rcv_layer, config.model.th.astype(
            np.float64), config.model.vs.astype(
            np.float64), config.receiver_distance.astype(
                np.float64))
    # * extract information from taup
    # first arrival array
    t0 = t0_vp
    # calculate the ray angle at the source
    dn, pa, sa = [np.zeros(len(config.receiver_distance),
                           dtype=np.float) for index in range(3)]
    # for each receiver, see calculate pa and sa
    for irec in range(len(config.receiver_distance)):
        if t0_vp[irec] < td_vp[irec] and p0_vp[irec] < 1. / 7:
            pa[irec] = config.model.vp[config.src_layer] * p0_vp[irec]
            dn[irec] = 1
        else:
            pa[irec] = config.model.vp[config.src_layer] * pd_vp[irec]
            dn[irec] = -1
        pa[irec] = np.rad2deg(np.arctan2(
            pa[irec], dn[irec] * np.sqrt(np.abs(1 - pa[irec] ** 2))))

        if t0_vs[irec] < td_vs[irec] and p0_vs[irec] < 1. / 4:
            sa[irec] = config.model.vs[config.src_layer] * p0_vs[irec]
            dn[irec] = 1
        else:
            sa[irec] = config.model.vs[config.src_layer] * pd_vs[irec]
            dn[irec] = -1
        sa[irec] = np.rad2deg(np.arctan2(
            sa[irec], dn[irec] * np.sqrt(np.abs(1 - sa[irec] ** 2))))

    # * if we should flip the model
    # get a copy of the earth model
    # ! note, we directly use model, src_layer, rcv_layer, as they might be fliped and we don't want to
    # ! change the config
    model = copy(config.model)
    src_layer = config.src_layer
    rcv_layer = config.rcv_layer
    flip: bool = False
    if rcv_layer > src_layer:
        flip = True
        src_layer = len(model.th) - src_layer
        rcv_layer = len(model.th) - rcv_layer
        # reverse the velocity model
        model.model_values = model.model_values[::-1, :]
    # for vs, it might be 0 in the sea, we assign a small value here
    model.model_values[:, 1][model.model_values[:, 1] < EPSILON] = EPSILON
    # get the source and receiver depth difference, the vs at source
    hs: float = 0.
    for index, value in enumerate(model.th):
        if rcv_layer <= index < src_layer:
            hs += value
    vs_source = model.vs[src_layer]

    # * calculate the si matrix representing source
    si = calculate_gf_source(config.source.srcType, model, flip, src_layer)

    # * initialize some parameters for waveform intergration
    dynamic = True
    nfft2 = int(config.npt / 2)
    wc1 = int(
        config.filter[0] * config.npt * config.dt) + 1
    wc2 = int(
        config.filter[1] * config.npt * config.dt) + 1
    if config.npt == 1:
        # it will never happen!
        dynamic = False
        nfft2 = 1
        wc1 = 1
    dw = np.pi * 2 / (config.npt * config.dt)
    sigma = SIGMA * dw / (np.pi * 2)
    wc = nfft2 * (1. - config.taper)
    if wc < 1:
        wc = 1
    else:
        wc = int(wc)
    # ! note, we will use taper, pmin, pmax, dk, sigma later
    taper = np.pi / (nfft2 - wc + 1)
    if wc2 > wc:
        wc2 = wc
    if wc1 > wc2:
        wc1 = wc2
    kc = config.kmax / hs
    pmin = config.pmin / vs_source
    pmax = config.pmax / vs_source
    xmax = np.max([hs, np.max(config.receiver_distance)])
    # update t0 based on number of samples before first arrival
    t0 -= config.samples_before_first_arrival * config.dt
    dk = config.dk * np.pi / xmax
    filter_const = dk / (np.pi * 2)
    # * main loop, calculate the green's function
    # * call the function from the cython module
    sum_waveform: np.ndarray = waveform_integration(
        model,
        config,
        src_layer,
        rcv_layer,
        taper,
        pmin,
        pmax,
        dk,
        nfft2,
        dw,
        kc,
        flip,
        filter_const,
        dynamic,
        wc1,
        wc2,
        t0,
        wc,
        si,
        sigma)
    # * with sum_waveform, we can apply the inverse fft acting as the frequency integration
    dt_smth = config.dt / config.smth
    nfft_smth = int(config.npt * config.smth)
    dfac = np.exp(sigma * dt_smth)
    if nfft2 == 1:
        static_return_list = []
        for irec in range(len(config.receiver_distance)):
            static_return_list.append(np.real(sum_waveform[irec, :, 0]))
        return static_return_list
    fac = np.array([dfac**index for index in range(nfft_smth)])
    nCom_mapper = {"dc": 9, "sf": 6, "ep": 3}
    nCom = nCom_mapper[config.source.srcType]

    # * do the ifftr
    gf_streamall = []
    # get correct t0 value
    for irec in range(len(config.receiver_distance)):
        stream_irec = Stream()
        for icom in range(nCom):
            waveform_freqdomain = np.hstack([sum_waveform[irec, icom, :], np.zeros(
                int(nfft_smth / 2) - nfft2, dtype=np.complex)])
            gf_data = irfft(waveform_freqdomain, nfft_smth) / dt_smth
            # now we apply the frequency correction
            fac_icom = fac * np.exp(sigma * t0[irec])
            gf_data = gf_data * fac_icom
            stats_sac = {
                "delta": dt_smth,
                "b": t0_vp[irec],
                "e": nfft_smth *
                dt_smth +
                t0_vp[irec],
                "o": 0.0,
                "dist": config.receiver_distance[irec],
                "t1": t0_vp[irec] +
                config.samples_before_first_arrival *
                config.dt,
                "t2": t0_vs[irec],
                "user1": pa[irec],
                "user2": sa[irec],
                "npts": nfft_smth,
            }
            trace_irec_icom = Trace(data=gf_data, header={
                "sac": stats_sac
            })
            trace_irec_icom.stats.starttime += t0_vp[irec]
            trace_irec_icom.stats.delta = dt_smth
            stream_irec += trace_irec_icom
        gf_streamall.append(stream_irec)

    # * here the green's function is gf_streamall
    return gf_streamall


def calculate_gf_source(
        src_type: str,
        model: SeisModel,
        flip: bool,
        src_layer: int) -> np.ndarray:
    """
    calculate the source matrix used in propogational matrix operations

    :param src_type: the type of the source
    :type src_type: str
    :param model: the Earth model
    :type model: SeisModel
    :param flip: if the source and the receiver should be fliped
    :type flip: bool
    :param src_layer: the layer where the source is located
    :type src_layer: int
    :return: the source matrix
    :rtype: np.ndarray
    """
    s: np.ndarray = np.zeros((3, 6), dtype=np.float)
    mu = model.rh * model.vs * model.vs
    xi = (model.vs ** 2) / (model.vp ** 2)
    if flip:
        flip_val = -1
    else:
        flip_val = 1
    if src_type == "dc":
        s[0, 1] = 2. * xi[src_layer] / mu[src_layer]
        s[0, 3] = 4. * xi[src_layer] - 3.
        s[1, 0] = flip_val / mu[src_layer]
        s[1, 4] = -s[1, 0]
        s[2, 3] = 1.
        s[2, 5] = -1.
    elif src_type == "ep":
        s[0, 1] = xi[src_layer] / mu[src_layer]
        s[0, 3] = 2. * xi[src_layer]
    elif src_type == "sf":
        s[0, 2] = -flip_val
        s[1, 3] = -1.
        s[1, 5] = 1.
    return s


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
        sigma: float) -> np.ndarray:
    """
    the main function wrapping the cython module, do the wave number integration.

    :param model: the Earth model
    :type model: SeisModel
    :param config: the configuration of calculating GF
    :type config: Config
    :param src_layer: the layer where the source is located (source at the bottom)
    :type src_layer: int
    :param rcv_layer: the layer where the receiver is located (receiver at the top)
    :type rcv_layer: int
    :param taper: the taper value
    :type taper: float
    :param pmin: the min slowness
    :type pmin: float
    :param pmax: the max slowness
    :type pmax: float
    :param dk: sampling interval of wavenumber
    :type dk: float
    :param nfft2: the half of simulation points
    :type nfft2: int
    :param dw: sampling interval of frequency
    :type dw: float
    :param kc: the max wave number
    :type kc: float
    :param flip: if the source and the receiver has been fliped
    :type flip: bool
    :param filter_const: the const value used in filtering
    :type filter_const: float
    :param dynamic: if performing the dynamic simulation
    :type dynamic: bool
    :param wc1: the starting point of wave number integration
    :type wc1: int
    :param wc2: the end point beyond which the filtering will take effect
    :type wc2: int
    :param t0: the first arrival time
    :type t0: np.ndarray
    :param wc: the end point beyond which the taper will take effect
    :type wc: int
    :param si: the source matrix
    :type si: np.ndarray
    :param sigma: the value to supress the numerical noise
    :type sigma: float
    :return: the result of doing wavenumber integration
    :rtype: np.ndarray
    """
    # * note, here we use parameters within config as their value is different from config
    mu = model.rh * model.vs * model.vs
    sum_waveform = np.zeros(
        (len(config.receiver_distance), 9, int(nfft2)), dtype=np.complex)
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
        sigma,
        sum_waveform)
    return sum_waveform
