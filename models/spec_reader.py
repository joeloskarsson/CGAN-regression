import os
from functools import partial

import constants
import utils

def read_spec(spec, spec_dir):
    path = os.path.join(spec_dir, "{}.json".format(spec))
    assert os.path.exists(path), (
            "Specification file '{}' does not exist".format(path))

    spec_dict = utils.json_to_dict(path)
    return spec_dict

read_cgan_spec = partial(read_spec, spec_dir=constants.CGAN_SPEC_DIR)
read_nn_spec = partial(read_spec, spec_dir=constants.NN_SPEC_DIR)

