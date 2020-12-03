from copy import copy
from typing import Optional, Tuple, Union

import numpy as np
from numpy.core.defchararray import mod
from numpy.lib.utils import source
from obspy.geodetics.base import degrees2kilometers
from pyfk.setting import R_EARTH
from pyfk.utils import PyfkError, PyfkWarning


class SeisModel(object):
    def __init__(self, model: Optional[np.ndarray] = None, flattening: bool = False, use_kappa: bool = False) -> None:
        if(not isinstance(model, np.ndarray)):
            raise PyfkError("Earth Model must be a 2D numpy array.")
        if(len(np.shape(model)) != 2):
            raise PyfkError("Earth Model must be a 2D numpy array.")

        self._flattening = flattening

        row: int
        column: int
        row, column = np.shape(model)
        if(column < 3):
            raise PyfkError(
                "Must provide at least three columns for the model")
        self.model_values: np.ndarray = np.zeros((row, 6), dtype=np.float)

        # * read model values and apply flattening
        fl: np.ndarray = np.ones(row, dtype=np.float)
        if(self._flattening):
            r = R_EARTH
            for irow in range(row):
                r = r-model[irow, 0]
                fl[irow] = R_EARTH/(r+0.5*model[irow, 0])
        self.model_values[:, 0] = model[:, 0]*fl
        # set the thickness for the last row as 0
        self.model_values[-1, 0] = 0.
        self.model_values[:, 1] = model[:, 1]*fl
        if(use_kappa):
            self.model_values[:, 2] = model[:, 2]*self.model_values[:, 1]*fl
        else:
            self.model_values[:, 2] = model[:, 2]*fl
        flag: bool = True
        if(column < 4 or np.any(model[:, 3] > 20.)):
            # cp qs to [:,4] and scale rho
            flag = False
            self.model_values[:, 4] = model[:, 3]
            self.model_values[:, 3] = 0.77+0.32*self.model_values[:, 2]
        else:
            self.model_values[:, 3] = model[:, 3]
        if(column < 5 and flag):
            self.model_values[:, 4] = 500.
        else:
            self.model_values[:, 4] = model[:, 4]
        if(column < 6):
            self.model_values[:, 5] = 2*self.model_values[:, 4]
        else:
            self.model_values[:, 5] = model[:, 5]

    @property
    def th(self) -> np.ndarray:
        return self.model_values[:, 0]

    @property
    def vs(self) -> np.ndarray:
        return self.model_values[:, 1]

    @property
    def vp(self) -> np.ndarray:
        return self.model_values[:, 2]

    @property
    def rh(self) -> np.ndarray:
        return self.model_values[:, 3]

    @property
    def qa(self) -> np.ndarray:
        return self.model_values[:, 4]

    @property
    def qb(self) -> np.ndarray:
        return self.model_values[:, 5]

    @property
    def flattening(self) -> bool:
        return self.flattening

    @flattening.setter
    def flattening(self, value: bool) -> None:
        self.flattening = value

    def add_layer(self, dd: float, idep: int) -> None:
        self.model_values = np.concatenate(
            (self.model_values, np.zeros((1, self.model_values.shape[1]))), axis=0)
        self.model_values[idep+1:, :] = self.model_values[idep:-1, :]
        self.model_values[idep, 0] -= dd
        # dd>0 or dd==0
        if(dd > 0):
            self.model_values[idep+1, 0] = dd

    def remove_topo(self) -> None:
        if(self.model_values[0, 0] < 0.):
            self.model_values[0, 0] = 0.

    def __copy__(self):
        new_instance = SeisModel(self.model_values.copy(), False, False)
        new_instance.flattening = self.flattening
        return new_instance


class SourceModel(object):
    def __init__(self, sdep: float = 0, srcType: str = "dc") -> None:
        self._sdep = sdep
        self._srcType = srcType

        if(self._sdep <= 0):
            raise PyfkError("Must use a positive source depth.")
        if(self._srcType not in ["dc", "sf", "ep"]):
            raise PyfkError(
                "Source type should be one of 'dc', 'sf', or 'ep'.")

    @property
    def sdep(self) -> float:
        return self._sdep

    @property
    def srcType(self) -> str:
        return self._srcType

    @sdep.setter
    def sdep(self, value: float) -> None:
        self._sdep = value


class Config(object):
    def __init__(self, model: Optional[SeisModel] = None, source: Optional[SourceModel] = None, receiver_distance: Optional[Union[list, np.ndarray]] = None, degrees: bool = False, taper: float = 0.3, filter: Tuple[float, float] = (0, 0), npt: int = 256, dt: float = 1.,
                 dk: float = 0.3, smth: float = 1., pmin: float = 0., pmax: float = 1., kmax: float = 15., rdep: float = 0., updn: str = "all", samples_before_first_arrival: int = 50) -> None:
        # * read in and validate parameters
        # receiver_distance
        if(receiver_distance == None):
            raise PyfkError("Must provide a list of receiver distance")
        self.receiver_distance: np.ndarray = np.array(receiver_distance)
        # degrees
        if(degrees):
            self.receiver_distance = np.array(
                list(map(degrees2kilometers, self.receiver_distance)))
        # taper
        if(taper <= 0 or taper > 1):
            raise PyfkError("Taper must be with (0,1)")
        self.taper = taper
        # filter
        if(filter[0] < 0 or filter[0] > 1 or filter[1] < 0 or filter[1] > 1):
            raise PyfkError(
                "Filter must be a tuple (f1,f2), f1 and f2 should be within (0,1)")
        self.filter = filter
        # npt
        if(npt <= 0):
            raise PyfkError("npt should be positive.")
        self.npt = npt
        # dt
        if(dt <= 0):
            raise PyfkError("dt should be positive.")
        self.dt = dt
        # dk
        if(dk <= 0 or dk >= 0.5):
            raise PyfkError("dk should be within (0,0.5)")
        if(dk <= 0.1 or dk >= 0.4):
            raise PyfkWarning("dk is recommended to be within (0.1,0.4)")
        self.dk = dk
        # smth
        if(smth <= 0):
            raise PyfkError("smth should be positive.")
        self.smth = smth
        # pmin
        if(pmin < 0 or pmin >= 1):
            raise PyfkError("pmin should be within (0,1)")
        self.pmin = pmin
        # pmax
        if(pmax < 0 or pmax >= 1):
            raise PyfkError("pmax should be within (0,1)")
        if(pmin >= pmax):
            raise PyfkError("pmin should be smaller than pmax")
        self.pmax = pmax
        # kmax
        if(kmax < 10):
            raise PyfkError("kmax should be larger or equal to 10")
        self.kmax = kmax
        # rdep
        if(rdep < 0):
            raise PyfkError("the receiver depth should be positive")
        self.rdep = rdep
        if(updn not in ["all", "up", "down"]):
            raise PyfkError(
                "the selection of phases should be either 'up', 'down' or 'all'")
        self.updn = updn
        if(samples_before_first_arrival <= 0):
            raise PyfkError("samples_before_first_arrival should be positive")
        self.samples_before_first_arrival = samples_before_first_arrival
        # source and model
        if(source == None):
            raise PyfkError("Must provide a source")
        if(model == None):
            raise PyfkError("Must provide a seisModel")
        self.source = source
        # use copy since the model will be modified
        self.model = copy(model)
        # * dealing with the coupling with the source and the seismodel
        if(self.model.flattening):
            self.source.sdep = self._flattening(self.source.sdep)
            self.rdep = self._flattening(self.rdep)

        # * get src_layer and rcv_layer
        free_surface: bool = self.model.th[0] > 0
        if(len(self.model.th) < 2):
            free_surface = True
        if(free_surface and (self.source.sdep < 0 or self.rdep < 0)):
            raise PyfkError("The source or receivers are located in the air.")
        if(self.source.sdep < self.rdep):
            self.src_layer = self._insert_intf(self.source.sdep)
            self.rcv_layer = self._insert_intf(self.rcv_layer)
        else:
            self.rcv_layer = self._insert_intf(self.rcv_layer)
            self.src_layer = self._insert_intf(self.source.sdep)
        # two src layers should have same vp
        if((self.model.vp[self.src_layer] != self.model.vp[self.src_layer-1]) or (self.model.vs[self.src_layer] != self.model.vs[self.src_layer-1])):
            raise PyfkError("The source is located at a real interface.")

    def _flattening(self, depth: float) -> float:
        return R_EARTH*np.log(R_EARTH/(R_EARTH-depth))

    def _insert_intf(self, depth: float) -> int:
        ndep = len(self.model.th)
        searching_depth: int = 0
        idep: int = 0
        for idep in range(ndep):
            searching_depth += self.model.th[idep]
            if(searching_depth > depth):
                break
        if((idep > 0 and depth == searching_depth-self.model.th[idep]) or (idep == 0 and depth == 0)):
            return idep
        dd = searching_depth-depth
        self.model.add_layer(dd, idep)
        if(self.model.th[0] < 0):
            self.source.sdep -= self.model.th[0]
            self.rdep -= self.model.th[0]
            self.model.remove_topo()
        return idep+1
