from typing import Union, Optional, List

import numpy as np
from obspy import Trace, Stream

from pyfk.config.config import SourceModel, Config
from pyfk.utils.error_message import PyfkError


def calculate_sync(gf: Union[List[Stream], Stream],
                   config: Config,
                   az: Union[float, int] = 0,
                   source_time_function: Optional[Trace] = None) -> List[Stream]:
    """
    Compute displacements in cm in the up, radial (outward), and transverse (clockwise) directions produced by different seismic sources

    :param gf: the Green's function list (or just an obspy Stream representing one distance) from calculate_gf; or read from the Green's function database from FK
    :type gf: Union[List[Stream], Stream]
    :param config: the configuration of sync calculation
    :type config: Config
    :param az: set station azimuth in degree measured from the North, defaults to 0
    :type az: Union[float, int], optional
    :param source_time_function: should be an obspy Trace with the data as the source time function, can use generate_source_time_function to generate a trapezoid shaped source time function, defaults to None
    :type source_time_function: Optional[Trace], optional
    :raises PyfkError: az must be a number
    :raises PyfkError: must provide a source time function
    :raises PyfkError: check input Green's function
    :raises PyfkError: delta for the source time function and the Green's function should be the same
    :return: a list of three components stream (ordered as z, r, t)
    :rtype: List[Stream]
    """
    # * handle input parameters
    if not (isinstance(az, float) or isinstance(az, int)):
        raise PyfkError("az must be a number")
    az = az % 360
    cmpaz_list = np.array([0., az, az + 90])
    if cmpaz_list[2] > 360.:
        cmpaz_list[2] -= 360.
    if isinstance(gf, Stream):
        gf = [gf]
    if source_time_function is None:
        raise PyfkError("must provide a source time function")
    if (not isinstance(gf, list)) or (len(gf) == 0) or (
            not isinstance(gf[0], Stream)):
        raise PyfkError("check input Green's function")
    for irec in range(len(gf)):
        for each_trace in gf[irec]:
            if each_trace.stats["delta"] != source_time_function.stats["delta"]:
                raise PyfkError(
                    "delta for the source time function and the Green's function should be the same")

    # * calculate the radiation pattern
    config.source.calculate_radiation_pattern(az)

    # * handle gf, project to the three component with amplitude from radiation pattern
    sync_gf = sync_calculate_gf(gf, config.source)

    # * the main loop to conv the source with the gf
    sync_result = []
    cmpinc_list = [0., 90., 90.]
    for irec in range(len(sync_gf)):
        sync_result.append(Stream())
        for icom in range(3):
            # do the convolution
            data_conv = np.convolve(
                source_time_function.data,
                sync_gf[irec][icom].data)[
                :len(
                    sync_gf[irec][icom].data)]
            header = {**sync_gf[irec][icom].stats}
            if "sac" not in header:
                # if called after gf, should never happen
                header["sac"] = {}
            header["sac"]["az"] = az
            header["sac"]["cmpinc"] = cmpinc_list[icom]
            header["sac"]["cmpaz"] = cmpaz_list[icom]
            sync_result[-1] += Trace(header=header, data=data_conv)
    return sync_result


def sync_calculate_gf(gf: List[Stream], source: SourceModel) -> List[Stream]:
    """
    stack three the GF into three components based on the radiation pattern

    :param gf: the calculated GF
    :type gf: List[Stream]
    :param source: the source model, with attached actual source
    :type source: SourceModel
    :return: the stacked GF
    :rtype: List[Stream]
    """
    npts: int = gf[0][0].stats["npts"]
    # * init the result, a list of 3 component stream
    nrec = len(gf)
    result = []
    for irec in range(nrec):
        result.append(Stream())
        for icom in range(3):
            thetrace = Trace(
                header=gf[irec][0].stats, data=np.zeros(npts, dtype=np.float))
            result[-1] += thetrace

    # * now we convert the gf to result
    for irec in range(nrec):
        for inn in range(source.nn):
            for icom in range(3):
                coef = source.m0
                if source.nn > 1:
                    coef *= source.rad[inn, icom]
                result[irec][icom].data += gf[irec][3 * inn + icom].data * coef
    return result


def generate_source_time_function(
        dura: float = 0.,
        rise: float = 0.5,
        delta: float = 0.1) -> Trace:
    """
    [summary]

    :param dura: [description], defaults to 0.
    :type dura: float, optional
    :param rise: [description], defaults to 0.5
    :type rise: float, optional
    :param delta: [description], defaults to 0.1
    :type delta: float, optional
    :return: [description]
    :rtype: Trace
    """
    ns = int(dura / delta)
    if ns < 2:
        ns = 2
    result_data = np.zeros(ns + 1, dtype=np.float)
    nr = int(rise * ns)
    if nr < 1:
        nr = 1
    if 2 * nr > ns:
        nr = ns / 2
    amp = 1. / (nr * (ns - nr))
    result_data[:nr] = amp * np.arange(nr)
    result_data[nr:ns - nr] = nr * amp
    result_data[ns - nr:] = (ns - np.arange(ns - nr, ns + 1)) * amp
    result_trace = Trace(header={}, data=result_data)
    result_trace.stats.delta = delta
    return result_trace
