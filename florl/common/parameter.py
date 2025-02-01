from typing import Any
from io import BytesIO

from flwr.common import Parameters
import torch
from torch.nn import Module

type StateDict = dict[str, Any]

TENSOR_TYPE = "torch.state_dict"


def get_torch_parameters(
    data: StateDict | Module, ignore_prefix: tuple[str, ...] = ()
) -> Parameters:
    """Converts a PyTorch state dict into a serialized Parameters object.

    Args:
        state (StateDict): A PyTorch state dict containing model parameters.

    Returns:
        Parameters: A Parameters object with the serialized state dict as bytes
                   and the tensor type set to "torch.state_dict".
    """
    if isinstance(data, dict):
        state_dict = data
    elif isinstance(data, Module):
        state_dict = data.state_dict()
    state_dict = _filter_out_prefix(state_dict, ignore_prefix)
    buffer = BytesIO()
    torch.save(obj=state_dict, f=buffer)
    return Parameters(tensors=[buffer.getvalue()], tensor_type=TENSOR_TYPE)


def load_torch_parameters(
    module: Module, parameters: Parameters, ignore_prefix: tuple[str, ...] = ()
) -> None:
    """Loads serialized PyTorch parameters into a given module.

    Args:
        module (nn.Module): The PyTorch module into which the parameters will be loaded.
        parameters (Parameters): A `Parameters` object containing the serialized state dictionary as a list of byte tensors.

    Raises:
        ValueError: If the `tensor_type` in `parameters` does not match the expected type torch.state_dict).
    """
    if parameters.tensor_type != TENSOR_TYPE:
        raise ValueError(
            f"Unexpected parameter type {parameters.tensor_type}, expected a {TENSOR_TYPE}"
        )
    buffer = BytesIO(initial_bytes=b"".join(parameters.tensors))
    buffer.seek(0)
    state_dict = _filter_out_prefix(
        torch.load(f=buffer, weights_only=True), ignore_prefix
    )
    module.load_state_dict(state_dict=state_dict, strict=False)


def _filter_out_prefix(state_dict: StateDict, prefix: tuple[str, ...]) -> StateDict:
    return {k: v for k, v in state_dict.items() if not k.startswith(prefix)}
