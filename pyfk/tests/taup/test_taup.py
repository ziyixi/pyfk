from os.path import dirname, join

import numpy as np
from obspy.taup import TauPyModel
from pyfk.taup.taup import taup

from pyfk.config.config import Config, SeisModel, SourceModel


class TestFunctionTaup(object):
    @staticmethod
    def gen_test_model(model_name):
        model_path = join(dirname(__file__), f"../data/{model_name}.nd")
        model_data_raw = np.loadtxt(model_path)
        # generate the model file used for taup
        len_interface = np.shape(model_data_raw)[0]
        model_data = np.zeros((len_interface - 1, 6), dtype=np.float)
        for index in range(len_interface - 1):
            model_data[index, 0] = model_data_raw[index + 1, 0] - \
                model_data_raw[index, 0]
            model_data[index, 1] = model_data_raw[index, 2]
            model_data[index, 2] = model_data_raw[index, 1]
            model_data[index, 3] = model_data_raw[index, 3]
            model_data[index, 4] = model_data_raw[index, 5]
            model_data[index, 5] = model_data_raw[index, 4]
        # remove the rows that thickness==0
        model_data = model_data[model_data[:, 0] > 0.05]
        return model_data

    def test_earth_models(self):
        for earth_model_name in ["prem", "ak135f_no_mud", "1066a"]:
            earthmodel = TauPyModel(model=earth_model_name)
            for source_depth in [12, 50, 200]:
                model_data = self.gen_test_model(earth_model_name)
                test_model = SeisModel(
                    model_data, flattening=True, use_kappa=False)
                test_source = SourceModel(sdep=source_depth, srcType="dc")
                receiver_distance = [1, 10, 50]
                test_config = Config(
                    model=test_model,
                    source=test_source,
                    receiver_distance=receiver_distance,
                    degrees=True)
                t0_list, _, _, _ = taup(
                    test_config.src_layer, test_config.rcv_layer, test_config.model.th.astype(
                        np.float64), test_config.model.vp.astype(
                        np.float64), test_config.receiver_distance.astype(
                        np.float64))
                for index, each_distance in enumerate(receiver_distance):
                    arrivals = earthmodel.get_travel_times(
                        source_depth_in_km=source_depth,
                        distance_in_degree=each_distance,
                        phase_list=[
                            "p",
                            "P"])
                    assert np.allclose(
                        arrivals[0].time, t0_list[index], rtol=0.01)
