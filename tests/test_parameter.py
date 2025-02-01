from pprint import pformat

from expecttest import assert_expected_inline
import torch
import torch.nn as nn

from florl.common import get_torch_parameters, load_model_parameters
from .util import initialise_parameters_to_float


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
