import numpy as np
import pytest
from obspy.geodetics.base import degrees2kilometers

from pyfk.config.config import Config, SeisModel, SourceModel
from pyfk.setting import R_EARTH
from pyfk.utils.error_message import PyfkError, PyfkWarning


class TestClassSeisModel(object):
    test_model_6column = np.array([
        [5.5, 3.18, 5.501, 2.53, 600, 1100],
        [10.5, 3.64, 6.301, 2.55, 700, 1300],
        [16.0, 3.87, 6.699, 2.59, 800, 1600],
        [90.0, 4.50, 7.799, 2.6, 900, 1800]
    ])

    def test_init_noflattening_nokappa_6column(self):
        test_model = SeisModel(
            TestClassSeisModel.test_model_6column,
            flattening=False,
            use_kappa=False)
        prefered_output = TestClassSeisModel.test_model_6column.copy()
        prefered_output[-1, 0] = 0
        assert np.allclose(test_model.model_values, prefered_output)

    def test_init_noflattening_kappa_6column(self):
        to_test = TestClassSeisModel.test_model_6column.copy()
        to_test[:, 2] = to_test[:, 2] / to_test[:, 1]
        test_model = SeisModel(
            to_test, flattening=False, use_kappa=True)
        prefered_output = TestClassSeisModel.test_model_6column.copy()
        prefered_output[-1, 0] = 0
        assert np.allclose(test_model.model_values, prefered_output)

    def test_init_flattening_nokappa_6column(self):
        test_model = SeisModel(
            TestClassSeisModel.test_model_6column,
            flattening=True,
            use_kappa=False)
        r = R_EARTH
        fl = np.ones(
            TestClassSeisModel.test_model_6column.shape[0], dtype=np.float)
        for irow in range(TestClassSeisModel.test_model_6column.shape[0]):
            r = r - TestClassSeisModel.test_model_6column[irow, 0]
            fl[irow] = R_EARTH / \
                (r + 0.5 * TestClassSeisModel.test_model_6column[irow, 0])
        prefered_output = TestClassSeisModel.test_model_6column.copy()
        prefered_output[-1, 0] = 0
        prefered_output[:, 0] *= fl
        prefered_output[:, 1] *= fl
        prefered_output[:, 2] *= fl
        assert np.allclose(test_model.model_values, prefered_output)

    def test_init_flattening_kappa_6column(self):
        to_test = TestClassSeisModel.test_model_6column.copy()
        to_test[:, 2] = to_test[:, 2] / to_test[:, 1]
        test_model = SeisModel(
            to_test, flattening=True, use_kappa=True)
        r = R_EARTH
        fl = np.ones(
            TestClassSeisModel.test_model_6column.shape[0], dtype=np.float)
        for irow in range(TestClassSeisModel.test_model_6column.shape[0]):
            r = r - TestClassSeisModel.test_model_6column[irow, 0]
            fl[irow] = R_EARTH / \
                (r + 0.5 * TestClassSeisModel.test_model_6column[irow, 0])
        prefered_output = TestClassSeisModel.test_model_6column.copy()
        prefered_output[-1, 0] = 0
        prefered_output[:, 0] *= fl
        prefered_output[:, 1] *= fl
        prefered_output[:, 2] *= fl
        assert np.allclose(test_model.model_values, prefered_output)

    def test_init_noflattening_nokappa_5column(self):
        test_model_5column = TestClassSeisModel.test_model_6column[:, :-1]
        test_model = SeisModel(
            test_model_5column, flattening=False, use_kappa=False)
        prefered_output = TestClassSeisModel.test_model_6column.copy()
        prefered_output[-1, 0] = 0
        prefered_output[:, -1] = prefered_output[:, -2] * 2
        assert np.allclose(test_model.model_values, prefered_output)

    def test_init_noflattening_nokappa_4column_normal(self):
        test_model_4column = TestClassSeisModel.test_model_6column[:, :-2]
        test_model = SeisModel(
            test_model_4column, flattening=False, use_kappa=False)
        prefered_output = TestClassSeisModel.test_model_6column.copy()
        prefered_output[-1, 0] = 0
        prefered_output[:, -1] = 1000.
        prefered_output[:, -2] = 500.
        assert np.allclose(test_model.model_values, prefered_output)

    def test_init_noflattening_nokappa_3column(self):
        # have rho in the column
        test_model_3column = TestClassSeisModel.test_model_6column[:, :-3]
        test_model = SeisModel(
            test_model_3column, flattening=False, use_kappa=False)
        prefered_output = TestClassSeisModel.test_model_6column.copy()
        prefered_output[-1, 0] = 0
        prefered_output[:, -1] = 1000.
        prefered_output[:, -2] = 500.
        prefered_output[:, -3] = 0.77 + 0.32 * prefered_output[:, 2]
        assert np.allclose(test_model.model_values, prefered_output)

    def test_init_noflattening_nokappa_4column_abnormal(self):
        # have rho in the column
        test_model_4column = TestClassSeisModel.test_model_6column[:, :-2].copy(
        )
        test_model_4column[:, 3] = TestClassSeisModel.test_model_6column[:, 4]
        test_model = SeisModel(
            test_model_4column, flattening=False, use_kappa=False)
        prefered_output = TestClassSeisModel.test_model_6column.copy()
        prefered_output[-1, 0] = 0
        prefered_output[:, 3] = 0.77 + 0.32 * prefered_output[:, 2]
        prefered_output[:, 5] = prefered_output[:, 4] * 2
        assert np.allclose(test_model.model_values, prefered_output)

    def test_get_attribute(self):
        test_model = SeisModel(
            TestClassSeisModel.test_model_6column,
            flattening=False,
            use_kappa=False)
        assert np.all(test_model.th[:-1] ==
                      TestClassSeisModel.test_model_6column[:-1, 0]) and test_model.th[-1] == 0
        assert np.all(test_model.vs ==
                      TestClassSeisModel.test_model_6column[:, 1])
        assert np.all(test_model.vp ==
                      TestClassSeisModel.test_model_6column[:, 2])
        assert np.all(test_model.rh ==
                      TestClassSeisModel.test_model_6column[:, 3])
        assert np.all(test_model.qs ==
                      TestClassSeisModel.test_model_6column[:, 4])
        assert np.all(test_model.qp ==
                      TestClassSeisModel.test_model_6column[:, 5])
        test_model.flattening = True
        assert test_model.flattening is True
        test_model.flattening = False

    def test_catch_exceptions(self):
        with pytest.raises(PyfkError) as execinfo:
            _ = SeisModel(
                TestClassSeisModel.test_model_6column[:, :2], flattening=False, use_kappa=False)
        assert str(
            execinfo.value) == 'Must provide at least three columns for the model'

        with pytest.raises(PyfkError) as execinfo:
            _ = SeisModel(
                TestClassSeisModel.test_model_6column.tolist(),
                flattening=False,
                use_kappa=False)
        assert str(
            execinfo.value) == 'Earth Model must be a 2D numpy array.'

        with pytest.raises(PyfkError) as execinfo:
            _ = SeisModel(
                TestClassSeisModel.test_model_6column.flatten(),
                flattening=False,
                use_kappa=False)
        assert str(
            execinfo.value) == 'Earth Model must be a 2D numpy array.'

    def test_copy(self):
        from copy import copy
        test_model = SeisModel(
            TestClassSeisModel.test_model_6column,
            flattening=False,
            use_kappa=False)
        copied_model = copy(test_model)
        assert np.all(copied_model.model_values == test_model.model_values)
        copied_model.model_values[:, :] += 1
        assert np.all(copied_model.model_values != test_model.model_values)

    def test_remove_topo(self):
        to_test = TestClassSeisModel.test_model_6column.copy()
        to_test[0, 0] = -to_test[0, 0]
        test_model = SeisModel(
            to_test, flattening=False, use_kappa=False)
        test_model.remove_topo()
        prefered_output = TestClassSeisModel.test_model_6column.copy()
        prefered_output[-1, 0] = 0
        prefered_output[0, 0] = 0
        assert np.allclose(test_model.model_values, prefered_output)

    def test_add_layer(self):
        test_model_6column_with_source = np.array([
            [5.5, 3.18, 5.501, 2.53, 600, 1100],
            [3, 3.64, 6.301, 2.55, 700, 1300],
            [7.5, 3.64, 6.301, 2.55, 700, 1300],
            [16.0, 3.87, 6.699, 2.59, 800, 1600],
            [0, 4.50, 7.799, 2.6, 900, 1800]
        ])
        test_model = SeisModel(
            TestClassSeisModel.test_model_6column,
            flattening=False,
            use_kappa=False)
        test_model.add_layer(7.5, 1)
        assert np.all(test_model.model_values ==
                      test_model_6column_with_source)


class TestClassSourceModel(object):
    def test_init(self):
        test_model = SourceModel(sdep=12, srcType="dc")
        assert test_model.sdep == 12
        assert test_model.srcType == "dc"

    def test_set_sdep(self):
        test_model = SourceModel(sdep=12, srcType="dc")
        test_model.sdep = 13
        assert test_model._sdep == 13
        assert test_model.sdep == 13

    def test_catch_exceptions(self):
        with pytest.raises(PyfkError) as execinfo:
            _ = SourceModel(sdep=12, srcType="bbq")
        assert str(
            execinfo.value) == "Source type should be one of 'dc', 'sf', or 'ep'."


class TestClassConfig(object):
    test_model_6column = np.array([
        [5.5, 3.18, 5.501, 2.53, 600, 1100],
        [10.5, 3.64, 6.301, 2.55, 700, 1300],
        [16.0, 3.87, 6.699, 2.59, 800, 1600],
        [90.0, 4.50, 7.799, 2.6, 900, 1800]
    ])

    def test_init(self):
        test_model = SeisModel(
            TestClassSeisModel.test_model_6column,
            flattening=False,
            use_kappa=False)
        test_source = SourceModel(sdep=12, srcType="dc")
        test_config = Config(
            model=test_model,
            source=test_source,
            receiver_distance=[
                10,
                20,
                30])
        newmodel = np.array([
            [5.5, 3.18, 5.501, 2.53, 600, 1100],
            [6.5, 3.64, 6.301, 2.55, 700, 1300],
            [4, 3.64, 6.301, 2.55, 700, 1300],
            [16.0, 3.87, 6.699, 2.59, 800, 1600],
            [0, 4.50, 7.799, 2.6, 900, 1800]
        ])
        assert np.all(test_config.model.model_values == newmodel)

        test_config = Config(
            model=test_model,
            source=test_source,
            receiver_distance=[
                10,
                20,
                30],
            rdep=16,
            degrees=True)
        receiver_distance_km = [degrees2kilometers(
            10), degrees2kilometers(20), degrees2kilometers(30)]
        assert np.allclose(test_config.receiver_distance, receiver_distance_km)

        test_config = Config(
            model=test_model,
            source=test_source,
            receiver_distance=[
                10,
                20,
                30],
            rdep=16)
        assert np.all(test_config.model.model_values == newmodel)

        test_config = Config(
            model=test_model,
            source=test_source,
            receiver_distance=[
                10,
                20,
                30],
            rdep=30)
        newmodel = np.array([
            [5.5, 3.18, 5.501, 2.53, 600, 1100],
            [6.5, 3.64, 6.301, 2.55, 700, 1300],
            [4, 3.64, 6.301, 2.55, 700, 1300],
            [14.0, 3.87, 6.699, 2.59, 800, 1600],
            [2.0, 3.87, 6.699, 2.59, 800, 1600],
            [0, 4.50, 7.799, 2.6, 900, 1800]
        ])
        assert np.all(test_config.model.model_values == newmodel)

        test_config = Config(
            model=test_model,
            source=test_source,
            receiver_distance=[
                10,
                20,
                30],
            rdep=13)
        newmodel = np.array([
            [5.5, 3.18, 5.501, 2.53, 600, 1100],
            [6.5, 3.64, 6.301, 2.55, 700, 1300],
            [1, 3.64, 6.301, 2.55, 700, 1300],
            [3, 3.64, 6.301, 2.55, 700, 1300],
            [16.0, 3.87, 6.699, 2.59, 800, 1600],
            [0, 4.50, 7.799, 2.6, 900, 1800]
        ])
        assert np.all(test_config.model.model_values == newmodel)

        test_config = Config(
            model=test_model,
            source=test_source,
            receiver_distance=[
                10,
                20,
                30],
            rdep=7)
        newmodel = np.array([
            [5.5, 3.18, 5.501, 2.53, 600, 1100],
            [1.5, 3.64, 6.301, 2.55, 700, 1300],
            [5, 3.64, 6.301, 2.55, 700, 1300],
            [4, 3.64, 6.301, 2.55, 700, 1300],
            [16.0, 3.87, 6.699, 2.59, 800, 1600],
            [0, 4.50, 7.799, 2.6, 900, 1800]
        ])
        assert np.all(test_config.model.model_values == newmodel)

    def test_catch_exceptions(self):
        test_model = SeisModel(
            TestClassSeisModel.test_model_6column,
            flattening=False,
            use_kappa=False)
        test_source = SourceModel(sdep=12, srcType="dc")
        # receiver_distance
        with pytest.raises(PyfkError) as execinfo:
            _ = Config(
                model=test_model, source=test_source)
        assert str(
            execinfo.value) == "Must provide a list of receiver distance"
        # taper
        with pytest.raises(PyfkError) as execinfo:
            _ = Config(
                model=test_model,
                source=test_source,
                receiver_distance=[
                    10,
                    20,
                    30],
                taper=-0.2)
        assert str(
            execinfo.value) == "Taper must be with (0,1)"
        # npt
        with pytest.raises(PyfkError) as execinfo:
            _ = Config(
                model=test_model,
                source=test_source,
                receiver_distance=[
                    10,
                    20,
                    30],
                npt=-4)
        assert str(
            execinfo.value) == "npt should be positive."
        # dt
        with pytest.raises(PyfkError) as execinfo:
            _ = Config(
                model=test_model,
                source=test_source,
                receiver_distance=[
                    10,
                    20,
                    30],
                dt=-0.4)
        assert str(
            execinfo.value) == "dt should be positive."
        # dk
        with pytest.raises(PyfkError) as execinfo:
            _ = Config(
                model=test_model,
                source=test_source,
                receiver_distance=[
                    10,
                    20,
                    30],
                dk=0.7)
        assert str(
            execinfo.value) == "dk should be within (0,0.5)"
        # smth
        with pytest.raises(PyfkError) as execinfo:
            _ = Config(
                model=test_model,
                source=test_source,
                receiver_distance=[
                    10,
                    20,
                    30],
                smth=0)
        assert str(
            execinfo.value) == "smth should be positive."
        # pmin
        with pytest.raises(PyfkError) as execinfo:
            _ = Config(
                model=test_model,
                source=test_source,
                receiver_distance=[
                    10,
                    20,
                    30],
                pmin=1.2)
        assert str(
            execinfo.value) == "pmin should be within [0,1]"
        # pmax
        with pytest.raises(PyfkError) as execinfo:
            _ = Config(
                model=test_model,
                source=test_source,
                receiver_distance=[
                    10,
                    20,
                    30],
                pmax=1.2)
        assert str(
            execinfo.value) == "pmax should be within [0,1]"
        with pytest.raises(PyfkError) as execinfo:
            _ = Config(
                model=test_model,
                source=test_source,
                receiver_distance=[
                    10,
                    20,
                    30],
                pmax=0.3,
                pmin=0.8)
        assert str(
            execinfo.value) == "pmin should be smaller than pmax"
        # kmax
        with pytest.raises(PyfkError) as execinfo:
            _ = Config(
                model=test_model,
                source=test_source,
                receiver_distance=[
                    10,
                    20,
                    30],
                kmax=1.2)
        assert str(
            execinfo.value) == "kmax should be larger or equal to 10"
        # updn
        with pytest.raises(PyfkError) as execinfo:
            _ = Config(
                model=test_model,
                source=test_source,
                receiver_distance=[
                    10,
                    20,
                    30],
                updn="bbq")
        assert str(
            execinfo.value) == "the selection of phases should be either 'up', 'down' or 'all'"
        # samples_before_first_arrival
        with pytest.raises(PyfkError) as execinfo:
            _ = Config(
                model=test_model,
                source=test_source,
                receiver_distance=[
                    10,
                    20,
                    30],
                samples_before_first_arrival=-
                12)
        assert str(
            execinfo.value) == "samples_before_first_arrival should be positive"
        # source and receiver
        with pytest.raises(PyfkError) as execinfo:
            _ = Config(
                source=test_source, receiver_distance=[10, 20, 30])
        assert str(
            execinfo.value) == "Must provide a seisModel"
        with pytest.raises(PyfkError) as execinfo:
            _ = Config(
                model=1, source=test_source, receiver_distance=[10, 20, 30])
        assert str(
            execinfo.value) == "Must provide a seisModel"
        with pytest.raises(PyfkError) as execinfo:
            _ = Config(
                model=test_model, receiver_distance=[10, 20, 30])
        assert str(
            execinfo.value) == "Must provide a source"
        with pytest.raises(PyfkError) as execinfo:
            _ = Config(
                source=1, model=test_model, receiver_distance=[10, 20, 30])
        assert str(
            execinfo.value) == "Must provide a source"
        # source located at real interface
        test_source_interface = SourceModel(sdep=16, srcType="dc")
        with pytest.raises(PyfkError) as execinfo:
            _ = Config(
                model=test_model,
                source=test_source_interface,
                receiver_distance=[
                    10,
                    20,
                    30])
        assert str(
            execinfo.value) == "The source is located at a real interface."

    def test_catch_warning(self):
        test_model = SeisModel(
            TestClassSeisModel.test_model_6column,
            flattening=False,
            use_kappa=False)
        test_source = SourceModel(sdep=12, srcType="dc")
        # dk
        with pytest.warns(PyfkWarning) as execinfo:
            _ = Config(
                model=test_model,
                source=test_source,
                receiver_distance=[
                    10,
                    20,
                    30],
                dk=0.05)
        assert len(execinfo[0].message.args) == 1
        assert execinfo[0].message.args[0] == "dk is recommended to be within (0.1,0.4)"

    def test__flattening(self):
        test_model = SeisModel(
            TestClassSeisModel.test_model_6column,
            flattening=True,
            use_kappa=False)
        test_source = SourceModel(sdep=12, srcType="dc")
        test_config = Config(
            model=test_model,
            source=test_source,
            receiver_distance=[
                10,
                20,
                30],
            rdep=22)
        assert test_config.source.sdep == R_EARTH * \
            np.log(R_EARTH / (R_EARTH - 12))
        assert test_config.rdep == R_EARTH * \
            np.log(R_EARTH / (R_EARTH - 22))

    def test_topo(self):
        test_model_data = np.array([
            [-5.5, 3.18, 5.501, 2.53, 600, 1100],
            [10.5, 3.64, 6.301, 2.55, 700, 1300],
            [16.0, 3.87, 6.699, 2.59, 800, 1600],
            [90.0, 4.50, 7.799, 2.6, 900, 1800]
        ])
        test_model = SeisModel(
            test_model_data, flattening=False, use_kappa=False)
        test_source = SourceModel(sdep=12, srcType="dc")
        test_config = Config(
            model=test_model,
            source=test_source,
            receiver_distance=[
                10,
                20,
                30])
        newmodel = np.array([
            [0., 3.18, 5.501, 2.53, 600, 1100],
            [5.5, 3.64, 6.301, 2.55, 700, 1300],
            [5, 3.64, 6.301, 2.55, 700, 1300],
            [7.0, 3.87, 6.699, 2.59, 800, 1600],
            [9.0, 3.87, 6.699, 2.59, 800, 1600],
            [0., 4.50, 7.799, 2.6, 900, 1800]
        ])
        assert np.all(test_config.model.model_values == newmodel)

    def test_free_surface(self):
        test_model_data = np.array([
            [10.5, 3.64, 6.301, 2.55, 700, 1300],
        ])
        test_model = SeisModel(
            test_model_data, flattening=False, use_kappa=False)
        test_source = SourceModel(sdep=-12, srcType="dc")
        with pytest.raises(PyfkError) as execinfo:
            _ = Config(
                model=test_model,
                source=test_source,
                receiver_distance=[
                    10,
                    20,
                    30])
        assert str(
            execinfo.value) == "The source or receivers are located in the air."

        test_model_data = np.array([
            [5.5, 3.18, 5.501, 2.53, 600, 1100],
            [10.5, 3.64, 6.301, 2.55, 700, 1300],
            [16.0, 3.87, 6.699, 2.59, 800, 1600],
            [90.0, 4.50, 7.799, 2.6, 900, 1800]
        ])
        test_model = SeisModel(
            test_model_data, flattening=False, use_kappa=False)
        test_source = SourceModel(sdep=-12, srcType="dc")
        with pytest.raises(PyfkError) as execinfo:
            _ = Config(
                model=test_model,
                source=test_source,
                receiver_distance=[
                    10,
                    20,
                    30])
        assert str(
            execinfo.value) == "The source or receivers are located in the air."

    def test_radiation_pattern(self):
        with pytest.raises(PyfkError) as execinfo:
            _ = SourceModel(sdep=12, srcType="dc", source_mechanism=[1])
        assert str(
            execinfo.value) == "length of source_mechanism is not correct"
        with pytest.raises(PyfkError) as execinfo:
            _ = SourceModel(sdep=12, srcType="sf", source_mechanism=[1, 1])
        assert str(
            execinfo.value) == "length of source_mechanism is not correct"
        with pytest.raises(PyfkError) as execinfo:
            _ = SourceModel(sdep=12, srcType="sf",
                            source_mechanism=[[1, 2], [3, 4]])
        assert str(
            execinfo.value) == "source_mechanism should be a 1D array"
        with pytest.raises(PyfkError) as execinfo:
            _ = SourceModel(sdep=12, srcType="sf",
                            source_mechanism=(1, 2, 3, 4))
        assert str(
            execinfo.value) == "source_mechanism must be None, a list or numpy.ndarray"
        with pytest.raises(PyfkError) as execinfo:
            test_source = SourceModel(sdep=12, srcType="sf")
            test_source.update_source_mechanism(None)
        assert str(
            execinfo.value) == "source mechanism couldn't be None"
        # * test the case of 1, 3, 4
        test_source = SourceModel(sdep=12, srcType="ep", source_mechanism=[1])
        test_source.calculate_radiation_pattern(30.)
        test_source = SourceModel(
            sdep=12,
            srcType="sf",
            source_mechanism=[
                1,
                1,
                1])
        test_source.calculate_radiation_pattern(30.)
        test_source = SourceModel(
            sdep=12, srcType="dc", source_mechanism=[
                1, 1, 1, 1])
        test_source.calculate_radiation_pattern(30.)
        test_source = SourceModel(
            sdep=12, srcType="dc", source_mechanism=[
                1, 1, 1, 1, 1, 1, 1])
