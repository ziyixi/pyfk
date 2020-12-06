from copy import copy
from typing import Optional, Tuple

import numpy as np
from obspy.core.trace import Trace
from pyfk.taup.taup import taup

from pyfk.config.config import Config, SeisModel
from pyfk.gf.ffr import ffr
from pyfk.gf.waveform_integration import waveform_integration
from pyfk.setting import EPSILON, SIGMA


def calculate_gf(config: Optional[Config] = None) -> None:
    # * firstly, we calculate the travel time and ray parameter for vp and vs
    t0_vp: np.ndarray
    td_vp: np.ndarray
    p0_vp: np.ndarray
    pd_vp: np.ndarray
    t0_vs: np.ndarray
    td_vs: np.ndarray
    p0_vs: np.ndarray
    pd_vs: np.ndarray
    t0_vp, td_vp, p0_vp, pd_vp = taup(config.src_layer, config.rcv_layer,
                                      config.model.th.astype(np.float64), config.model.vp.astype(np.float64),
                                      config.receiver_distance.astype(np.float64))
    t0_vs, td_vs, p0_vs, pd_vs = taup(config.src_layer, config.rcv_layer,
                                      config.model.th.astype(np.float64), config.model.vs.astype(np.float64),
                                      config.receiver_distance.astype(np.float64))
    # * extract information from taup
    # first arrival
    t0 = t0_vp
    # calculate the ray angle at the source
    dn, pa, sa = np.zeros(len(config.receiver_distance), dtype=np.float)
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
    si = calculate_gf_source(config.source.srcType, model, flip)

    # * initialize some parameters for waveform intergration
    dynamic = True
    nfft2 = int(config.npt / 2)
    wc1 = int(
        config.filter[0] * config.npt * config.dt) + 1
    wc2 = int(
        config.filter[1] * config.npt * config.dt) + 1
    if config.npt == 1:
        dynamic = False
        nfft2 = 1
        wc1 = 1
    dw = np.pi * 2 / (config.npt * config.dt)
    sigma = SIGMA * dw / (np.pi * 2)
    wc = int(nfft2 * (1. - config.taper))
    if wc < 1:
        wc = 1
    # ! note, we will use taper, pmin, pmax, dk later
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

    # * main loop, calculate the green function
    numerical_integration_sum: np.ndarray = waveform_integration()
    gf_data_all = ffr(numerical_integration_sum)
    gf_head = generate_gf_head()
    # ! to be continued


def calculate_gf_source(src_type: str, model: SeisModel, flip: bool) -> np.ndarray:
    s: np.ndarray = np.zeros((3, 6), dtype=np.float)
    mu = model.rh * model.vs * model.vs
    xi = (model.vs ** 2) / (model.vp ** 2)
    if flip:
        flip_val = -1
    else:
        flip_val = 1
    if src_type == "dc":
        s[0, 1] = 2. * xi / mu
        s[0, 3] = 4. * xi - 3.
        s[1, 0] = flip_val / mu
        s[1, 4] = -s[1, 0]
        s[2, 3] = 1.
        s[2, 5] = -1.
    elif src_type == "ep":
        s[0, 1] = xi / mu
        s[0, 3] = 2. * xi
    elif src_type == "sf":
        s[0, 2] = -flip_val
        s[1, 3] = -1.
        s[1, 5] = 1.
    return s


def generate_gf_head():
    pass


class GF(object):
    def __init__(self, gf_data: Tuple[np.ndarray, ...], gf_head: dict) -> None:
        # * the gf will be a simple list, and its number should be determined by src_type
        self._gf = []
        for each_gf_data in gf_data:
            self._gf.append(Trace(data=each_gf_data, header=gf_head))

    @property
    def gf(self):
        return self._gf
