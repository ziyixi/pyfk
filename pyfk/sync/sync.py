from pyfk.config.config import SourceModel, Config
from typing import Union, Optional, List
from obspy import Trace, Stream
import numpy as np
from pyfk.utils.error_message import PyfkError


def calculate_sync(gf: Union[List[Stream], Stream],
                   config: Config,
                   az: Union[float, int] = 0,
                   source_time_function: Optional[Trace] = None):
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
    for irec in range(len(gf)):
        for each_trace in gf[irec]:
            if each_trace.stats["npts"] != source_time_function.stats["npts"]:
                raise PyfkError(
                    "npts for the source time function and the Green's function should be the same")

    # * handle gf, project to the three component with amplitude from radiation pattern
    sync_gf = sync_calculate_gf(gf, config.source)

    # * the main loop to conv the source with the gf
    sync_result = []
    cmpinc_list = [0., 90., 90.]
    for irec in range(len(sync_gf)):
        sync_result.append(Stream())
        for icom in range(3):
            # do the convolution
            data_conv = np.conv(
                source_time_function.data,
                sync_gf[irec][icom].data)[
                :len(
                    sync_gf[irec][icom].data)]
            header = {**sync_gf[irec][icom].stats}
            if "sac" not in header:
                header["sac"] = {}
            header["sac"]["az"] = az
            header["sac"]["cmpinc"] = cmpinc_list[icom]
            header["sac"]["cmpaz"] = cmpaz_list[icom]
            sync_result[-1] += Trace(header=header, data=data_conv)
    return sync_result


def sync_calculate_gf(gf: List[Stream], source: SourceModel):
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
