from pprint import pformat

from expecttest import assert_expected_inline
import numpy as np
import torch
import torch.nn as nn

from florl.common import (
    get_torch_parameters,
    load_model_parameters,
    torch_to_numpy,
    numpy_to_torch,
)
from florl._dev.tests import initialise_parameters_to_float


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 3)
        self.fc2 = nn.Linear(3, 1)

    def forward(self, x):
        return self.fc2(self.fc1(x))


def test_torch_parameters():
    model = Model()
    parameters = get_torch_parameters(model)

    # Verify tensor type
    assert_expected_inline(parameters.tensor_type, """torch.state_dict""")

    # Verify tensors
    assert len(parameters.tensors) == 1
    reference_state_dict = model.state_dict()
    load_model_parameters(model, parameters)
    for k, v in model.state_dict().items():
        assert k in reference_state_dict
        assert torch.equal(v, reference_state_dict[k])


def test_filter_parameters():
    model = Model()
    initialise_parameters_to_float(model, 1.0)
    assert_expected_inline(
        pformat(model.state_dict()),
        """\
OrderedDict([('fc1.weight',
              tensor([[1., 1.],
        [1., 1.],
        [1., 1.]])),
             ('fc1.bias', tensor([1., 1., 1.])),
             ('fc2.weight', tensor([[1., 1., 1.]])),
             ('fc2.bias', tensor([1.]))])""",
    )
    model_load = Model()
    initialise_parameters_to_float(model_load, 0.0)

    load_model_parameters(
        model, get_torch_parameters(model_load), ignore_prefix=("fc1",)
    )

    assert_expected_inline(
        pformat(model.state_dict()),
        """\
OrderedDict([('fc1.weight',
              tensor([[1., 1.],
        [1., 1.],
        [1., 1.]])),
             ('fc1.bias', tensor([1., 1., 1.])),
             ('fc2.weight', tensor([[0., 0., 0.]])),
             ('fc2.bias', tensor([0.]))])""",
    )


def test_torch_numpy_conversion():
    model = Model()
    state_dict = model.state_dict()
    weights = torch_to_numpy(state_dict)

    for (_, tensor), numpy_array in zip(sorted(state_dict.items()), weights):
        assert np.array_equal(numpy_array, tensor.cpu().numpy())

    state_dict["should_ignore"] = None
    weights_comparison = torch_to_numpy(state_dict)
    assert all((np.array_equal(x, y) for x, y in zip(weights, weights_comparison)))

    model.load_state_dict(
        numpy_to_torch(weights_comparison, state_dict, inplace=False), strict=False
    )

    for (_, tensor), numpy_array in zip(sorted(state_dict.items()), weights):
        assert np.array_equal(numpy_array, tensor.cpu().numpy())
