from os.path import dirname, join

import numpy as np
import obspy

from pyfk.config.config import Config, SeisModel, SourceModel
from pyfk.gf.gf import calculate_gf
from pyfk.tests.taup.test_taup import TestFunctionTaup


class TestFunctioncalculateGf(object):
    @staticmethod
    def test_hk():
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

    def test_big_array(self):
        # model_data = TestFunctionTaup.gen_test_model("prem")
        # there is a possibility that we write x=f(x) where x is a memoryview in the code
        # this might cause segmentation fault
        model_data = np.loadtxt(join(dirname(__file__), f"../data/hk"))
        model_hk = SeisModel(model=model_data)
        source_hk = SourceModel(sdep=16.5)
        config_hk = Config(
            model=model_hk,
            source=source_hk,
            npt=512,
            dt=0.1,
            receiver_distance=np.arange(10, 40, 10))
        _ = calculate_gf(config_hk)

    def test_prem_ep(self):
        model_data = TestFunctionTaup.gen_test_model("prem")
        model_prem = SeisModel(model=model_data)
        source_prem = SourceModel(sdep=16.5, srcType="ep")
        config_prem = Config(
            model=model_prem,
            source=source_prem,
            npt=512,
            dt=5,
            receiver_distance=[50])
        gf = calculate_gf(config_prem)
        for index, comnname in enumerate(["a", "b", "c"]):
            gf_data = obspy.read(
                join(
                    dirname(__file__),
                    f"../data/sync_prem_ep/50.grn.{comnname}"))[0]
            coef = np.corrcoef(
                gf_data.data,
                gf[0][index].data,
            )[0, 1]
            if np.isnan(coef):
                coef = 1.
            assert coef > 0.99

    def test_prem_sf(self):
        model_data = TestFunctionTaup.gen_test_model("prem")
        model_prem = SeisModel(model=model_data)
        source_prem = SourceModel(sdep=16.5, srcType="sf")
        config_prem = Config(
            model=model_prem,
            source=source_prem,
            npt=512,
            dt=1,
            receiver_distance=[50])
        gf = calculate_gf(config_prem)
        for index, comnname in enumerate(range(6)):
            gf_data = obspy.read(
                join(
                    dirname(__file__),
                    f"../data/sync_prem_sf/50.grn.{comnname}"))[0]
            coef = np.corrcoef(
                gf_data.data,
                gf[0][index].data,
            )[0, 1]
            if np.isnan(coef):
                coef = 1.
            assert coef > 0.99

    def test_receiver_deeper(self):
        model_data = TestFunctionTaup.gen_test_model("prem")
        model_prem = SeisModel(model=model_data)
        source_prem = SourceModel(sdep=16.5, srcType="dc")
        config_prem = Config(
            model=model_prem,
            source=source_prem,
            npt=512,
            dt=1,
            receiver_distance=[50],
            rdep=20)
        gf = calculate_gf(config_prem)
        for index, comnname in enumerate(range(9)):
            gf_data = obspy.read(
                join(
                    dirname(__file__),
                    f"../data/sync_receiver_deeper/50_20.grn.{comnname}"))[0]
            coef = np.corrcoef(
                gf_data.data,
                gf[0][index].data,
            )[0, 1]
            if np.isnan(coef):
                coef = 1.
            assert coef > 0.99

    def test_static_source(self):
        model_data = TestFunctionTaup.gen_test_model("prem")
        model_prem = SeisModel(model=model_data)
        source_prem = SourceModel(sdep=16.5, srcType="dc")
        config_prem = Config(
            model=model_prem,
            source=source_prem,
            npt=1,
            dt=1,
            receiver_distance=[50])
        gf = calculate_gf(config_prem)
        ref_gf = [-0.242E-06, -0.103E-05, 0.000E+00, 0.236E-06,
                  0.118E-05, -0.548E-07, -0.942E-07, -0.156E-05, 0.285E-06]
        coef = np.corrcoef(
            gf,
            ref_gf,
        )[0, 1]
        assert coef > 0.99999

    def test_smth(self):
        model_data = TestFunctionTaup.gen_test_model("prem")
        model_prem = SeisModel(model=model_data)
        source_prem = SourceModel(sdep=16.5, srcType="dc")
        config_prem = Config(
            model=model_prem,
            source=source_prem,
            npt=512,
            dt=0.1,
            smth=8,
            receiver_distance=[50])
        gf = calculate_gf(config_prem)
        for index, comnname in enumerate(range(9)):
            gf_data = obspy.read(
                join(
                    dirname(__file__),
                    f"../data/sync_smth/50.grn.{comnname}"))[0]
            coef = np.corrcoef(
                gf_data.data,
                gf[0][index].data,
            )[0, 1]
            if np.isnan(coef):
                coef = 1.
            assert coef > 0.99

    def test_filter(self):
        model_data = TestFunctionTaup.gen_test_model("prem")
        model_prem = SeisModel(model=model_data)
        source_prem = SourceModel(sdep=16.5, srcType="dc")
        config_prem = Config(
            model=model_prem,
            source=source_prem,
            npt=512,
            dt=0.1,
            smth=8,
            filter=(0.1, 0.6),
            receiver_distance=[50])
        gf = calculate_gf(config_prem)
        for index, comnname in enumerate(range(9)):
            gf_data = obspy.read(
                join(
                    dirname(__file__),
                    f"../data/sync_filter/50.grn.{comnname}"))[0]
            coef = np.corrcoef(
                gf_data.data,
                gf[0][index].data,
            )[0, 1]
            if np.isnan(coef):
                coef = 1.
            assert coef > 0.99
