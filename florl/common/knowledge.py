from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import struct
from typing import Dict, List

from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from florl.common.util import get_torch_parameters, set_torch_parameters
from flwr.common.typing import (
    Code,
    Status,
    GetParametersIns,
    GetParametersRes,
    Parameters,
    NDArrays,
)
from torch import nn


@dataclass
class KnowledgeShard:
    """Represents a subset of parameters within Knowledge"""

    # TODO: Should correspond to gRPC directly instead of serialisation hack
    # TODO: Should name also be serialised into parameter?

    id_: int
    name: str
    parameters: Parameters | None

    def pack(self) -> Parameters:
        """Represents the Serialised Parameter representation of a Knowledge Shard

        Returns:
            Parameters: Serialised representation.
        """
        # Validation
        if self.parameters is None:
            raise ValueError("Parameters is none.")
        if self.id_ < 0 or self.id_ > 65535:
            raise ValueError(f"id {self.id_} cannot be serialised to 2 bytes")
        if "." in self.name:
            raise ValueError("Do not include char . in name")

        # Serialisation
        packed_id = struct.pack(">I", self.id_)
        return Parameters(
            tensors=[packed_id + tensor for tensor in self.parameters.tensors],
            tensor_type=f"{self.name}.{self.parameters.tensor_type}",
        )

    @staticmethod
    def unpack(parameters: Parameters) -> KnowledgeShard:
        name, tensor_type = parameters.tensor_type.split(".", maxsplit=1)
        id_ = struct.unpack(">I", parameters.tensors[0][:4])[0]
        tensors = list(map(lambda x: x[4:], parameters.tensors))
        return KnowledgeShard(
            id_=id_,
            name=name,
            parameters=Parameters(tensors=tensors, tensor_type=tensor_type),
        )


class Knowledge(ABC):
    """Represents abstract information used in an algorithm"""

    # TODO: Serialisation hack
    SEP_CHAR = "|"

    def __init__(self, shards: List[str]) -> None:
        """Defines Knowledge

        Args:
            modules (List[str]): Names of each knowledge shard within knowledge
        """
        super().__init__()
        self._shards_registry = {
            name: KnowledgeShard(id_=i, name=name, parameters=None)
            for i, name in enumerate(shards)
        }

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        """Return the parameter serialisation of this Knowledge

        Args:
            ins (GetParametersIns): Config can contain "shards" - a list of shard names to fetch

        Returns:
            GetParametersRes: Parameter results.
        """
        # List of shards to fetch
        shards: List[str] = ins.config.get(
            "shards", [shard.name for shard in self._shards_registry.values()]
        )
        shards = [self._shards_registry[s] for s in shards]
        # Fetch parameters
        all_parameters_res = [self.get_shard_parameters(s) for s in shards]
            # Status Validation
        for parameter_res in all_parameters_res:
            if parameter_res.status.code != Code.OK:
                return parameter_res
            
        parameters = Parameters(
            tensor_type=Knowledge.SEP_CHAR.join(
                [x.parameters.tensor_type for x in all_parameters_res]
            ),
            tensors=sum([x.parameters.tensors for x in all_parameters_res], []),
        )
        # Combination Serialisation
        return GetParametersRes(
            status=Status(
                code=Code.OK,
                message="",
            ),
            parameters=parameters
        )
    
    def set_parameters(self, ins: Parameters) -> None:
        """ Set the parameters for knowledge

        Args:
            ins (Parameters): From Knowledge.get_parameters

        Raises:
            ValueError: Inconsistent parameter mapping
        """
        # Deserialisation
        i = 0
        tensor_types = ins.tensor_type.split(Knowledge.SEP_CHAR)
        tensor_buffer = []
        previous_id = None

        def process_set_buffer(i: int, tensor_buffer: List[bytes]) -> None:
            shard = KnowledgeShard.unpack(parameters=Parameters(
                tensors=tensor_buffer,
                tensor_type=tensor_types[i]
            ))
            if self._shards_registry[shard.name].id_ != shard.id_:
                raise ValueError(f"Inconsistent ID name mapping for {shard.name}")
            self._shards_registry[shard.name] = shard
            return shard

        for id_tensor in ins.tensors:
            id_ = struct.unpack(">I", id_tensor[:4])[0]
            if previous_id is None or previous_id == id_:
                tensor_buffer.append(id_tensor)
            else:
                # Set and callback
                self._set_shard_callback(process_set_buffer(i, tensor_buffer))
                # Reset state
                i, tensor_buffer = i + 1, [id_tensor]
            previous_id = id_

        if len(tensor_buffer) > 0:
            self._set_shard_callback(process_set_buffer(i, tensor_buffer))

    def get_shard_parameters(self, shard: KnowledgeShard) -> GetParametersRes:
        """ Fetches parameters from a shard

        Returns:
            GetParametersRes: fetch result.
        """
        if shard.name not in self._shards_registry:
            return GetParametersRes(
                Status(
                    code=Code.GET_PARAMETERS_NOT_IMPLEMENTED,
                    message=f"Module {shard.name} does not exist in registry.",
                )
            )
        parameters = self._get_shard_parameters(shard.name)
        shard.parameters = parameters
        return GetParametersRes(
            status=Status(Code.OK, ""),
            parameters=shard.pack()
        )

    # ==========
    # Implement these!
    # ==========

    @abstractmethod
    def _get_shard_parameters(self, name: str) -> Parameters:
        raise NotImplementedError

    @abstractmethod
    def _set_shard_callback(self, shard: KnowledgeShard) -> None:
        raise NotImplementedError

class NumPyKnowledge(Knowledge, ABC):
    """ Knowledge, where each shard has a corresponding NumPy parameter representation.
    """
    def _get_shard_parameters(self, name: str) -> Parameters:
        numpy_parameters = self._get_shard_parameters_numpy(name)
        return ndarrays_to_parameters(numpy_parameters)

    @property
    def torch_modules_registry(self) -> Dict[str, nn.Module]:
        """ Optional registry of mappings from names to Torch network
        """
        return {}

    def _set_shard_callback(self, shard: KnowledgeShard) -> None:
        return self._set_shard_callback_numpy(shard.name, parameters_to_ndarrays(shard.parameters))

    def _get_shard_parameters_numpy(self, name: str) -> NDArrays:
        if name in self.torch_modules_registry:
            return get_torch_parameters(self.torch_modules_registry[name])
        raise NotImplementedError

    def _set_shard_callback_numpy(self, name: str, params: NDArrays) -> None:
        if name in self.torch_modules_registry:
            return set_torch_parameters(
                self.torch_modules_registry[name],
                params
            )
        raise NotImplementedError
