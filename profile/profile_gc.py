from os.path import dirname, join

import numpy as np
from pyfk.gf.gf import calculate_gf
from pyfk.config.config import Config, SeisModel, SourceModel
from pyfk.tests.taup.test_taup import TestFunctionTaup


def main():
    model_data = TestFunctionTaup.gen_test_model("prem")
    # model_data = np.loadtxt(join(dirname(__file__), f"../pyfk/tests/data/hk"))
    model_hk = SeisModel(model=model_data)
    source_hk = SourceModel(sdep=16.5)
    config_hk = Config(
        model=model_hk,
        source=source_hk,
        npt=512,
        dt=0.1,
        receiver_distance=np.arange(10, 110, 10))
    _ = calculate_gf(config_hk)


if __name__ == '__main__':
    main()
