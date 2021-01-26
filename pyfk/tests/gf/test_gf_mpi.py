from os.path import dirname, join

import numpy as np
import obspy
import pytest
from pyfk import mpi_info
from pyfk.config.config import Config, SeisModel, SourceModel
from pyfk.gf.gf import calculate_gf


class TestFunctioncalculateGf_MPI(object):
    @pytest.mark.mpi
    def test_mpi_info(self):
        assert mpi_info().startswith("MPI installed correctly")

    @pytest.mark.mpi
    def test_val_correct(self):
        # test if the result of using only a single core equal to the mpirun result
        # * the same as test_gf
        # * perl fk.pl -Mhk/15/k -N512/0.1 10 20 30
        model_path = join(dirname(__file__), f"../data/hk")
        model_data = np.loadtxt(model_path)
        model_hk = SeisModel(model=model_data, use_kappa=True)
        source_hk = SourceModel(sdep=15)
        config_hk = Config(
            model=model_hk,
            source=source_hk,
            npt=512,
            dt=0.1,
            receiver_distance=[
                10,
                20,
                30])

        result = calculate_gf(config_hk)
        # * for all the gf in data/hk_gf, test if the results are close (in FK, it uses float but we are using double)
        for irec, each_rec in enumerate([10, 20, 30]):
            for icomn in range(9):
                hk_gf_data = obspy.read(
                    join(
                        dirname(__file__),
                        f"../data/hk_gf/{each_rec}.grn.{icomn}"))[0]
                coef = np.corrcoef(
                    hk_gf_data.data,
                    result[irec][icomn].data,
                )[0, 1]
                if np.isnan(coef):
                    coef = 1.0
                assert coef > 0.99
