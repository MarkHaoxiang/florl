from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
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
        if "." in self.name:
            raise ValueError("Do not include char . in name")

        # Serialisation
        return Parameters(
            tensors=self.parameters.tensors,
            tensor_type=f"{self.name}.{len(self.parameters.tensors)}.{self.parameters.tensor_type}",
        )

    @staticmethod
    def unpack(parameters: Parameters) -> KnowledgeShard:
        name, length, tensor_type = parameters.tensor_type.split(".", maxsplit=2)
        tensors = parameters.tensors
        if int(length) != len(tensors):
            raise ValueError(
                f"Inconsistency in tensors length for {name} with lengths {length} and {len(tensors)}"
            )
        return KnowledgeShard(
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
            name: KnowledgeShard(name=name, parameters=None) for name in shards
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
            parameters=parameters,
        )

    @staticmethod
    def unpack(ins: Parameters) -> List[KnowledgeShard]:
        """Unpacks parameters into shards

        Args:
            ins (Parameters): serialised knowledge.

        Returns:
            List[KnowledgeShard]: deserialisation.
        """
        # Deserialisation
        tensor_types = ins.tensor_type.split(Knowledge.SEP_CHAR)
        remaining = ins.tensors
        contained_shards = []
        for tensor_type in tensor_types:
            _, length, _ = tensor_type.split(".", maxsplit=2)
            length = int(length)
            tensors, remaining = remaining[:length], remaining[length:]
            shard = KnowledgeShard.unpack(
                parameters=Parameters(tensors=tensors, tensor_type=tensor_type)
            )
            contained_shards.append(shard)
        return contained_shards

    def update_knowledge(
        self, shards: List[KnowledgeShard], shard_filter: List[str] | None = None
    ) -> None:
        for shard in shards:
            if shard_filter is None or shard.name in shard_filter:
                self._shards_registry[shard.name] = shard
                self._set_shard_callback(shard)

    def set_parameters(self, ins: Parameters, shards: List[str] | None = None) -> None:
        """Set the parameters for knowledge

        Args:
            ins (Parameters): From Knowledge.get_parameters.
            shards (List[str] | None, optional): List of shards to synchronise. Defaults to all shards.
        """
        # Deserialisation
        contained_shards = Knowledge.unpack(ins)
        self.update_knowledge(contained_shards, shard_filter=shards)

    def get_shard_parameters(self, shard: KnowledgeShard) -> GetParametersRes:
        """Fetches parameters from a shard

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
        return GetParametersRes(status=Status(Code.OK, ""), parameters=shard.pack())

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
    """Knowledge, where each shard has a corresponding NumPy parameter representation."""

    def _get_shard_parameters(self, name: str) -> Parameters:
        numpy_parameters = self._get_shard_parameters_numpy(name)
        return ndarrays_to_parameters(numpy_parameters)

    @property
    def torch_modules_registry(self) -> Dict[str, nn.Module]:
        """Optional registry of mappings from names to Torch network"""
        return {}

    def _set_shard_callback(self, shard: KnowledgeShard) -> None:
        return self._set_shard_callback_numpy(
            shard.name, parameters_to_ndarrays(shard.parameters)
        )

    def _get_shard_parameters_numpy(self, name: str) -> NDArrays:
        if name in self.torch_modules_registry:
            return get_torch_parameters(self.torch_modules_registry[name])
        raise NotImplementedError

    def _set_shard_callback_numpy(self, name: str, params: NDArrays) -> None:
        if name in self.torch_modules_registry:
            return set_torch_parameters(self.torch_modules_registry[name], params)
        raise NotImplementedError
