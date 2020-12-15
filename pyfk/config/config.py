import warnings
from copy import copy
from typing import Optional, Tuple, Union

import numpy as np
from obspy.core.event.event import Event
from obspy.core.event.source import Tensor
from obspy.geodetics.base import degrees2kilometers

from pyfk.config.radiats import dc_radiat, sf_radiat, mt_radiat
from pyfk.setting import R_EARTH
from pyfk.utils.error_message import PyfkError, PyfkWarning


class SeisModel(object):
    def __init__(self,
                 model: np.ndarray = None,
                 flattening: bool = False,
                 use_kappa: bool = False) -> None:
        """
        SeisModel stores the Earth model

        :param model: numpy.ndarray model: a 2D numpy array storing the information of the 1D earth model, with the same format as FK.
               model has the following format (in units of km, km/s, g/cm3, each column):
               thickness vs vp_or_vp/vs [rho Qs Qp]
               rho=0.77 + 0.32*vp if not provided or the 4th column is larger than 20 (treated as Qs).
               Qs=500, Qp=2*Qs, if they are not specified.
               If the first layer thickness is zero, it represents the top elastic half-space.
               Otherwise, the top half-space is assumed to be vacuum and does not need to be specified.
               The last layer (i.e. the bottom half space) thickness should be always be zero. (if not, will use 0 anyway), defaults to None
        :type model: np.ndarray
        :param flattening: if the model and the source should be flatten, defaults to False
        :type flattening: bool, optional
        :param use_kappa: if the third column of the model file is vp/vs ratio, defaults to False
        :type use_kappa: bool, optional
        :raises PyfkError: Earth Model must be a 2D numpy array
        :raises PyfkError: Must provide at least three columns for the model
        """
        if not isinstance(model, np.ndarray):
            raise PyfkError("Earth Model must be a 2D numpy array.")
        if len(np.shape(model)) != 2:
            raise PyfkError("Earth Model must be a 2D numpy array.")

        self._flattening = flattening

        row: int
        column: int
        row, column = np.shape(model)
        if column < 3:
            raise PyfkError(
                "Must provide at least three columns for the model")
        self.model_values: np.ndarray = np.zeros((row, 6), dtype=np.float)
        # * read model values and apply flattening
        fl: np.ndarray = np.ones(row, dtype=np.float)
        if self._flattening:
            r = R_EARTH
            for irow in range(row):
                r = r - model[irow, 0]
                fl[irow] = R_EARTH / (r + 0.5 * model[irow, 0])
        self.model_values[:, 0] = model[:, 0] * fl
        # set the thickness for the last row as 0
        self.model_values[-1, 0] = 0.
        self.model_values[:, 1] = model[:, 1] * fl
        if use_kappa:
            # already have fl in self.model_values[:, 1]
            self.model_values[:, 2] = model[:, 2] * self.model_values[:, 1]
        else:
            self.model_values[:, 2] = model[:, 2] * fl
        # * handle other columns
        if column == 3:
            self.model_values[:, 3] = 0.77 + 0.32 * self.model_values[:, 2]
            self.model_values[:, 4] = 500.
            self.model_values[:, 5] = 2 * self.model_values[:, 4]
        elif column == 4:
            if np.any(model[:, 3] > 20.):
                self.model_values[:, 3] = 0.77 + 0.32 * self.model_values[:, 2]
                self.model_values[:, 4] = model[:, 3]
                self.model_values[:, 5] = 2 * self.model_values[:, 4]
            else:
                self.model_values[:, 3] = model[:, 3]
                self.model_values[:, 4] = 500.
                self.model_values[:, 5] = 2 * self.model_values[:, 4]
        elif column == 5:
            self.model_values[:, 3:5] = model[:, 3:5]
            self.model_values[:, 5] = 2 * self.model_values[:, 4]
        elif column == 6:
            self.model_values[:, 3:] = model[:, 3:]

    @property
    def th(self) -> np.ndarray:
        """
        get the thickness array

        :return: the thickness array
        :rtype: np.ndarray
        """
        return self.model_values[:, 0]

    @property
    def vs(self) -> np.ndarray:
        """
        get the vs array

        :return: the vs array
        :rtype: np.ndarray
        """
        return self.model_values[:, 1]

    @property
    def vp(self) -> np.ndarray:
        """
        get the vp array

        :return: the vp array
        :rtype: np.ndarray
        """
        return self.model_values[:, 2]

    @property
    def rh(self) -> np.ndarray:
        """
        get the density array

        :return: the density array
        :rtype: np.ndarray
        """
        return self.model_values[:, 3]

    @property
    def qs(self) -> np.ndarray:
        """
        get the attenuation qs array

        :return: the attenuation qs array
        :rtype: np.ndarray
        """
        return self.model_values[:, 4]

    @property
    def qp(self) -> np.ndarray:
        """
        get the attenuation qp array

        :return: the attenuation qp array
        :rtype: np.ndarray
        """
        return self.model_values[:, 5]

    @property
    def flattening(self) -> bool:
        """
        get the flatten status of the model

        :return: the flatten status of the model
        :rtype: bool
        """
        return self._flattening

    @flattening.setter
    def flattening(self, value: bool) -> None:
        """
        set the flatten status of the model

        :param value: the desired flatten value
        :type value: bool
        """
        self._flattening = value

    def add_layer(self, dd: float, idep: int) -> None:
        """
        insert a layer so that the source/receiver will be located at the interface of the layers

        :param dd: the distance(km) to the bottom of the later where the source/receiver is located
        :type dd: float
        :param idep: the layer (top most layer as index 0) where the source/receiver is located
        :type idep: int
        """
        self.model_values = np.concatenate(
            (self.model_values, np.zeros(
                (1, self.model_values.shape[1]))), axis=0)
        self.model_values[idep + 1:, :] = self.model_values[idep:-1, :]
        self.model_values[idep, 0] -= dd
        # dd>0 or dd==0
        if dd > 0:
            self.model_values[idep + 1, 0] = dd

    def remove_topo(self) -> None:
        """
        make the thickness for the first later to be 0 when the thickness is smaller than 0.
        """
        if self.model_values[0, 0] < 0.:
            self.model_values[0, 0] = 0.

    def __copy__(self):
        """
        make a copy of the current model, and the model numpy file will not be influenced

        :return: the copied SeisModel
        :rtype: SeisModel
        """
        new_instance = SeisModel(self.model_values.copy(), False, False)
        new_instance.flattening = self.flattening
        return new_instance

    def __repr__(self) -> str:
        return f"SeisModel(layers={np.shape(self.model_values)[0]}, flattening={self.flattening})"


class SourceModel(object):
    def __init__(self, sdep: float = 0., srcType: str = "dc",
                 source_mechanism: Optional[Union[list, np.ndarray]] = None) -> None:
        """
        the information about the the source used in the FK

        :param sdep: the depth of the source (km), and it should not be located between the interfaces of the Earth model, defaults to 0
        :type sdep: float
        :param srcType: the source type, can be ep (explosion), sf (single force) or dc (double couple), defaults to "dc"
        :type srcType: str, optional
        :param source_mechanism: a list with length of 1 (ep), 3 (sf), 4 or 7 (dc) with the same order as FK, or a Event of obspy (can read CMT solution file using obspy.read_events), defaults to None
        :type source_mechanism: Optional[Union[list, np.ndarray]], optional
        :raises PyfkError: [description]
        """
        self._sdep = sdep
        self._srcType = srcType

        if self._srcType not in ["dc", "sf", "ep"]:
            raise PyfkError(
                "Source type should be one of 'dc', 'sf', or 'ep'.")

        self._source_mechanism = None
        self._update_source_mechanism(source_mechanism)

    @property
    def sdep(self) -> float:
        """
        get the source depth in km

        :return: the source depth in km
        :rtype: float
        """
        return self._sdep

    @property
    def srcType(self) -> str:
        """
        get the source type (ep, sf or dc)

        :return: the source type (ep, sf or dc)
        :rtype: str
        """
        return self._srcType

    @sdep.setter
    def sdep(self, value: float) -> None:
        """
        set the depth for the source model

        :param value: the depth for the source model
        :type value: float
        """
        self._sdep = value

    @property
    def nn(self) -> int:
        """
        get the value associated with the source type

        :return: the value associated with the source type
        :rtype: int
        """
        return self._nn

    @property
    def m0(self) -> float:
        """
        get the value associated with the magnitude in FK

        :return: the value associated with the magnitude in FK
        :rtype: float
        """
        return self._m0

    @property
    def rad(self) -> np.ndarray:
        """
        get the radiation pattern

        :return: the radiation pattern
        :rtype: np.ndarray
        """
        return self._rad

    def calculate_radiation_pattern(self, az: float) -> None:
        """
        calculate the radiation pattern 3*3 array

        :param az: station azimuth in degree measured from the North (clockwise)
        :type az: float
        :raises PyfkError: length of source_mechanism must be 1, 3, 4, 7
        """
        mt = np.zeros((3, 3))
        if len(self._source_mechanism) == 1:
            self._m0 = self._source_mechanism[0] * 1e-20
            self._nn = 1
            self._rad = None
        elif len(self._source_mechanism) == 3:
            self._m0 = self._source_mechanism[0] * 1e-15
            self._nn = 2
            mt[0, :2] = self._source_mechanism[1:3]
            self._rad = sf_radiat(az - mt[0, 0], mt[0, 1])
        elif len(self._source_mechanism) == 4:
            self._m0 = np.power(
                10., 1.5 * self._source_mechanism[0] + 16.1 - 20)
            self._nn = 3
            mt[0, :] = self._source_mechanism[1:]
            self._rad = dc_radiat(az - mt[0, 0], mt[0, 1], mt[0, 2])
        elif len(self._source_mechanism) == 7:
            self._m0 = self._source_mechanism[0] * 1e-20
            self._nn = 3
            mt[0, :] = self._source_mechanism[1:4]
            mt[1, 1:] = self._source_mechanism[4:6]
            mt[2, 2] = self._source_mechanism[6]
            self._rad = mt_radiat(az, mt)
        else:
            # actually will never satisfied in real case
            raise PyfkError("length of source_mechanism must be 1, 3, 4, 7")

    def _update_source_mechanism(
            self, source_mechanism: Optional[Union[list, np.ndarray]]):
        """
        the internal function to update the source mechanism information

        :param source_mechanism: a list with length of 1 (ep), 3 (sf), 4 or 7 (dc) with the same order as FK, or a Event of obspy (can read CMT solution file using obspy.read_events)
        :type source_mechanism: Optional[Union[list, np.ndarray]]
        :raises PyfkError: source_mechanism should be a 1D array
        :raises PyfkError: length of source_mechanism is not correct
        :raises PyfkError: source_mechanism must be None, a list or numpy.ndarray
        """
        self._source_mechanism: Optional[np.ndarray]
        if isinstance(
                source_mechanism,
                list) or isinstance(
                source_mechanism,
                np.ndarray):
            if len(np.shape(source_mechanism)) != 1:
                raise PyfkError("source_mechanism should be a 1D array")
            typemapper = {
                "dc": [4, 7],
                "sf": [3],
                "ep": [1]
            }
            if len(source_mechanism) not in typemapper[self._srcType]:
                raise PyfkError("length of source_mechanism is not correct")
            self._source_mechanism = np.array(source_mechanism)
        elif isinstance(source_mechanism, Event):
            tensor: Tensor = source_mechanism.focal_mechanisms[0].moment_tensor.tensor
            # convert the tensor in RTP(USE) to NED, refer to
            # https://gfzpublic.gfz-potsdam.de/rest/items/item_272892/component/file_541895/content
            # page4
            # * the reason why we mul 1e-19 here is to keep the value the same as global cmt website standard format
            m_zz = tensor.m_rr * 1e-19
            m_xx = tensor.m_tt * 1e-19
            m_yy = tensor.m_pp * 1e-19
            m_xz = tensor.m_rt * 1e-19
            m_yz = -tensor.m_rp * 1e-19
            m_xy = -tensor.m_tp * 1e-19
            m0 = source_mechanism.focal_mechanisms[0].moment_tensor.scalar_moment * 1e7
            self._source_mechanism = np.array(
                [m0, m_xx, m_xy, m_xz, m_yy, m_yz, m_zz])
        elif source_mechanism is None:
            self._source_mechanism = None
        else:
            raise PyfkError(
                "source_mechanism must be None, a list or numpy.ndarray")

    def update_source_mechanism(
            self, source_mechanism: Union[list, np.ndarray, Event]):
        """
        update the source mechanism information after creation of the SourceModel

        :param source_mechanism: a list with length of 1 (ep), 3 (sf), 4 or 7 (dc) with the same order as FK, or a Event of obspy (can read CMT solution file using obspy.read_events)
        :type source_mechanism: Union[list, np.ndarray, Event]
        :raises PyfkError: source mechanism couldn't be None
        """
        if source_mechanism is None:
            raise PyfkError("source mechanism couldn't be None")
        self._update_source_mechanism(source_mechanism)

    def __repr__(self) -> str:
        return f"SourceModel(sdep={self.sdep}, srcType={self.srcType}, source_mechanism={self._source_mechanism})"


class Config(object):
    def __init__(self,
                 model: Optional[SeisModel] = None,
                 source: Optional[SourceModel] = None,
                 receiver_distance: Optional[Union[list,
                                                   np.ndarray]] = None,
                 degrees: bool = False,
                 taper: float = 0.3,
                 filter: Tuple[float,
                               float] = (0,
                                         0),
                 npt: int = 256,
                 dt: float = 1.,
                 dk: float = 0.3,
                 smth: float = 1.,
                 pmin: float = 0.,
                 pmax: float = 1.,
                 kmax: float = 15.,
                 rdep: float = 0.,
                 updn: str = "all",
                 samples_before_first_arrival: int = 50) -> None:
        """
        The configuration class used in generating Green's function and the synthetic waveform.

        :param model: the Earth model used in calculation, defaults to None
        :type model: Optional[SeisModel]
        :param source: the source model used in calculation, defaults to None
        :type source: Optional[SourceModel]
        :param receiver_distance: a list of receiver distance in km, defaults to None
        :type receiver_distance: Optional[Union[list, np.ndarray]]
        :param degrees: use degrees instead of km, defaults to False
        :type degrees: bool, optional
        :param taper: taper applies a low-pass cosine filter at fc=(1-taper)*f_Niquest, defaults to 0.3
        :type taper: float, optional
        :param filter: apply a high-pass filter with a cosine transition zone between freq. f1 and f2 in Hz, defaults to (0, 0)
        :type filter: Tuple[float, float], optional
        :param npt: the number of points, defaults to 256
        :type npt: int, optional
        :param dt: the sampling interval in seconds, defaults to 1.
        :type dt: float, optional
        :param dk: the non-dimensional sampling interval of wavenumber, defaults to 0.3
        :type dk: float, optional
        :param smth: makes the final sampling interval to be dt/smth, defaults to 1.
        :type smth: float, optional
        :param pmin: the min slownesses in term of 1/vs_at_the_source, defaults to 0.
        :type pmin: float, optional
        :param pmax: the max slownesses in term of 1/vs_at_the_source, defaults to 1.
        :type pmax: float, optional
        :param kmax: kmax at zero frequency in term of 1/hs, defaults to 15.
        :type kmax: float, optional
        :param rdep: the depth for the receivers in km, defaults to 0.
        :type rdep: float, optional
        :param updn: "up" for up-going wave only, "down" for down-going wave only, "all" for both "up" and "down", defaults to "all"
        :type updn: str, optional
        :param samples_before_first_arrival: the number of points before the first arrival, defaults to 50
        :type samples_before_first_arrival: int, optional
        :raises PyfkError: Must provide a list of receiver distance
        :raises PyfkError: Taper must be with (0,1)
        :raises PyfkError: Filter must be a tuple (f1,f2), f1 and f2 should be within [0,1]
        :raises PyfkError: npt should be positive.
        :raises PyfkError: dt should be positive.
        :raises PyfkError: dk should be within (0,0.5)
        :raises PyfkError: smth should be positive.
        :raises PyfkError: pmin should be within [0,1]
        :raises PyfkError: pmax should be within [0,1]
        :raises PyfkError: pmin should be smaller than pmax
        :raises PyfkError: kmax should be larger or equal to 10
        :raises PyfkError: the selection of phases should be either 'up', 'down' or 'all'
        :raises PyfkError: samples_before_first_arrival should be positive
        :raises PyfkError: Must provide a source
        :raises PyfkError: Must provide a seisModel
        """
        # * read in and validate parameters
        # receiver_distance
        if receiver_distance is None:
            raise PyfkError("Must provide a list of receiver distance")
        self.receiver_distance: np.ndarray = np.array(
            receiver_distance, dtype=np.float64)
        # degrees
        if degrees:
            self.receiver_distance = np.array(
                list(map(degrees2kilometers, self.receiver_distance)))
        # taper
        if taper <= 0 or taper > 1:
            raise PyfkError("Taper must be with (0,1)")
        self.taper = taper
        # filter
        self.filter = filter
        # npt
        if npt <= 0:
            raise PyfkError("npt should be positive.")
        self.npt = npt
        if self.npt == 1:
            # we don't use st_fk
            self.npt = 2
        # dt
        if dt <= 0:
            raise PyfkError("dt should be positive.")
        if self.npt == 2 and dt < 1000:
            self.dt = 1000
        else:
            self.dt = dt
        # dk
        if dk <= 0 or dk >= 0.5:
            raise PyfkError("dk should be within (0,0.5)")
        if dk <= 0.1 or dk >= 0.4:
            warnings.warn(PyfkWarning(
                "dk is recommended to be within (0.1,0.4)"))
        self.dk = dk
        # smth
        if smth <= 0:
            raise PyfkError("smth should be positive.")
        self.smth = smth
        # pmin
        if pmin < 0 or pmin > 1:
            raise PyfkError("pmin should be within [0,1]")
        self.pmin = pmin
        # pmax
        if pmax < 0 or pmax > 1:
            raise PyfkError("pmax should be within [0,1]")
        if pmin >= pmax:
            raise PyfkError("pmin should be smaller than pmax")
        self.pmax = pmax
        # kmax
        if kmax < 10:
            raise PyfkError("kmax should be larger or equal to 10")
        self.kmax = kmax
        # rdep
        self.rdep = rdep
        # updn
        if updn not in ["all", "up", "down"]:
            raise PyfkError(
                "the selection of phases should be either 'up', 'down' or 'all'")
        self.updn = updn
        # samples_before_first_arrival
        if samples_before_first_arrival <= 0:
            raise PyfkError("samples_before_first_arrival should be positive")
        self.samples_before_first_arrival = samples_before_first_arrival
        # source and model
        if (source is None) or (not isinstance(source, SourceModel)):
            raise PyfkError("Must provide a source")
        if (model is None) or (not isinstance(model, SeisModel)):
            raise PyfkError("Must provide a seisModel")
        self.source = source
        # use copy since the model will be modified
        self.model = copy(model)
        self._couple_model_and_source()

    def _couple_model_and_source(self) -> None:
        # * dealing with the coupling with the source and the seismodel
        if self.model.flattening:
            self.source.sdep = self._flattening_func(self.source.sdep)
            self.rdep = self._flattening_func(self.rdep)

        # * get src_layer and rcv_layer
        free_surface: bool = self.model.th[0] > 0
        if len(self.model.th) < 2:
            free_surface = True
        if free_surface and (self.source.sdep < 0 or self.rdep < 0):
            raise PyfkError("The source or receivers are located in the air.")
        if self.source.sdep < self.rdep:
            self.src_layer = self._insert_intf(self.source.sdep)
            self.rcv_layer = self._insert_intf(self.rdep)
        else:
            self.rcv_layer = self._insert_intf(self.rdep)
            self.src_layer = self._insert_intf(self.source.sdep)
        # two src layers should have same vp
        if (self.model.vp[self.src_layer] != self.model.vp[self.src_layer - 1]) or (
                self.model.vs[self.src_layer] != self.model.vs[self.src_layer - 1]):
            raise PyfkError("The source is located at a real interface.")

    @staticmethod
    def _flattening_func(depth: float) -> float:
        return R_EARTH * np.log(R_EARTH / (R_EARTH - depth))

    def _insert_intf(self, depth: float) -> int:
        ndep = len(self.model.th)
        searching_depth: int = 0
        idep: int = 0
        for idep in range(ndep):
            searching_depth += self.model.th[idep]
            if searching_depth > depth:
                break
        if (idep > 0 and depth == searching_depth -
                self.model.th[idep]) or (idep == 0 and depth == 0):
            return idep
        dd = searching_depth - depth
        self.model.add_layer(dd, idep)
        if self.model.th[0] < 0:
            self.source.sdep -= self.model.th[0]
            self.rdep -= self.model.th[0]
            self.model.remove_topo()
        return idep + 1

    def __repr__(self) -> str:
        return f"Config(model={self.model.__repr__()}, source={self.source.__repr__()}, receiver_distance={self.receiver_distance}, taper={self.taper}, filter={self.filter}, npt={self.npt}, dt={self.dt}, dk={self.dk}, smth={self.smth}, pmin={self.pmin}, pmax={self.pmax}, kmax={self.kmax}, rdep={self.rdep}, updn={self.updn}, samples_before_first_arrival={self.samples_before_first_arrival})"
