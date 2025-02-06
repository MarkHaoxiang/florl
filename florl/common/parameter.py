from typing import Any
from collections import OrderedDict
from io import BytesIO

from flwr.common import Parameters, ndarrays_to_parameters, parameters_to_ndarrays
import numpy as np
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
    else:
        raise ValueError()
    state_dict = _filter_out_prefix(state_dict, ignore_prefix)
    buffer = BytesIO()
    torch.save(obj=state_dict, f=buffer)
    return Parameters(tensors=[buffer.getvalue()], tensor_type=TENSOR_TYPE)


def load_torch_parameters(
    parameters: Parameters, ignore_prefix: tuple[str, ...] = ()
) -> StateDict:
    """Deserialises a Parameters object to obtain the state_dict.

    Args:
        parameters (Parameters): A `Parameters` object containing serialized tensor data.
        ignore_prefix (tuple[str, ...], optional): A tuple of key prefixes to exclude from
            the resulting state dictionary. Defaults to an empty tuple.

    Raises:
        ValueError: If the `Parameters` object contains an unexpected tensor type.

    Returns:
        StateDict: A PyTorch state dictionary containing the deserialized parameters,
                  excluding any keys that match the `ignore_prefix`.
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
    return state_dict


def load_model_parameters(
    model: Module,
    data: Parameters | StateDict,
    ignore_prefix: tuple[str, ...] = (),
) -> None:
    """Loads serialized PyTorch parameters into a given module.

    Args:
        module (nn.Module): The PyTorch module into which the parameters will be loaded.
        parameters (Parameters): A `Parameters` object containing the serialized state dictionary as a list of byte tensors.

    Raises:
        ValueError: If the `tensor_type` in `parameters` does not match the expected type torch.state_dict).
    """
    if isinstance(data, dict):
        state_dict = data
    elif isinstance(data, Parameters):
        state_dict = load_torch_parameters(data)
    else:
        raise ValueError()
    state_dict = _filter_out_prefix(state_dict, ignore_prefix)
    model.load_state_dict(state_dict=state_dict, strict=False)


def torch_to_numpy(state_dict: StateDict) -> list[np.ndarray]:
    """Convert a PyTorch state dictionary to a list of NumPy ndarrays.

    Args:
        state_dict (StateDict): A PyTorch state dictionary containing model parameters.

    Returns:
        list[np.ndarray]: A list of NumPy ndarrays, where each array corresponds to a tensor
                          in the state dictionary, sorted by the keys (parameter names).
    """
    return [
        v.numpy(force=True)
        for _, v in sorted(state_dict.items(), key=lambda x: x[0])
        if isinstance(v, torch.Tensor)
    ]


def numpy_to_torch(
    weights: list[np.ndarray], reference: StateDict, inplace: bool = True
) -> StateDict:
    """Convert a list of NumPy ndarrays back into a PyTorch state dictionary.

    This function updates the provided state dictionary with the values from the list of
    NumPy ndarrays. The keys in the state dictionary are preserved, and the values are
    replaced with the corresponding tensors converted from NumPy arrays.

    Args:
        weights (list[np.ndarray]): A list of NumPy ndarrays to convert to PyTorch tensors.
        state_dict (StateDict): The original PyTorch state dictionary whose values will be updated.
        inplace (bool, optional): If True, updates the state dictionary in place. If False,
                                  returns a new state dictionary. Defaults to True.

    Returns:
        StateDict: The updated state dictionary with values replaced by PyTorch tensors.
                  If `inplace` is True, the original state dictionary is modified and returned.
                  If `inplace` is False, a new state dictionary is returned.
    """
    weights_keys = [k for k, v in reference.items() if isinstance(v, torch.Tensor)]
    weights_original: list[torch.Tensor] = [reference.get(k) for k in weights_keys]  # type: ignore
    weights_keys.sort()

    if len(weights_keys) != len(weights):
        raise ValueError(
            f"Length mismatch: {len(weights)} weights provided, but state_dict has {len(weights_keys)} tensors."
        )

    reference
    weights_dict = {
        k: torch.from_numpy(v).to(device=o.device, dtype=o.dtype)
        for k, v, o in zip(weights_keys, weights, weights_original)
    }

    if inplace:
        reference.update(weights_dict)
        return reference
    else:
        return OrderedDict({k: weights_dict.get(k, v) for k, v in reference.items()})


def torch_to_numpy_parameters(parameters: Parameters) -> Parameters:
    """Convert PyTorch parameters into a `Parameters` object containing NumPy ndarrays.

    This is an utility to provide compatability with Flower strategies, which by default
     aggregate numpy arrays.

    Args:
        parameters (Parameters): A `Parameters` object containing a PyTorch state dict.

    Returns:
        Parameters: A `Parameters` object where the PyTorch tensors have been replaced
                    with their corresponding NumPy ndarrays.
    """
    state_dict = load_torch_parameters(parameters)
    parameters = ndarrays_to_parameters(torch_to_numpy(state_dict))
    return parameters


def numpy_to_torch_parameters(
    parameters: Parameters, reference: StateDict
) -> Parameters:
    """Convert a `Parameters` object containing NumPy ndarrays back into a PyTorch state dictionary.

    Args:
        parameters (Parameters): A `Parameters` object containing NumPy ndarrays.
        reference (StateDict): The original PyTorch state dictionary whose values will be updated.

    Returns:
        Parameters: A `Parameters` object with PyTorch state_dict
    """
    weights = parameters_to_ndarrays(parameters)
    updated_state_dict = numpy_to_torch(weights, reference, inplace=False)
    return get_torch_parameters(updated_state_dict)


def _filter_out_prefix(state_dict: StateDict, prefix: tuple[str, ...]) -> StateDict:
    return {k: v for k, v in state_dict.items() if not k.startswith(prefix)}
