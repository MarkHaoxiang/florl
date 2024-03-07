from typing import List
from flwr.common.typing import GetParametersIns, Parameters
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
import numpy as np

from florl.common.knowledge import KnowledgeShard, Knowledge


rng = np.random.default_rng(seed=0)


class InjectionTestKnowledge(Knowledge):
    def __init__(self, default_value: np.ndarray, shards: List[str]) -> None:
        super().__init__(shards)
        self._shard_values = {
            shard.name: default_value for shard in self._shards_registry.values()
        }

    def _get_shard_parameters(self, name: str) -> Parameters:
        return ndarrays_to_parameters(self._shard_values[name])

    def _set_shard_callback(self, shard: KnowledgeShard) -> None:
        self._shard_values[shard.name] = parameters_to_ndarrays(shard.parameters)


class TestKnowledge:
    """Test the knowledge data abstraction"""

    def test_knowledge_shard(self):
        # Build shard
        test_parameters = rng.random(
            size=(
                2,
                3,
            )
        )
        shard = KnowledgeShard(
            name="test_shard", parameters=ndarrays_to_parameters(test_parameters)
        )
        # Pack
        serialised = shard.pack()
        # Unpack
        shard_ = KnowledgeShard.unpack(serialised)
        # Comparison check
        result_parameters = parameters_to_ndarrays(shard_.parameters)
        assert (result_parameters == test_parameters).all(), "Unpacked arrays differs"

    def test_knowledge(self):
        knowl_1 = InjectionTestKnowledge(
            default_value=np.array([1]), shards=[str(i) for i in range(3)]
        )
        parameters_res = knowl_1.get_parameters(GetParametersIns({}))
        assert (
            parameters_res.parameters.tensor_type
            == "0.1.numpy.ndarray|1.1.numpy.ndarray|2.1.numpy.ndarray"
        )

        knowl_2 = InjectionTestKnowledge(
            default_value=np.array([2]), shards=[str(i) for i in range(5)]
        )
        knowl_2.set_parameters(parameters_res.parameters)

        for name in knowl_2._shard_values.keys():
            if name in knowl_1._shard_values:
                assert (
                    knowl_2._shard_values[name] == knowl_1._shard_values[name]
                ).all()
            else:
                assert (knowl_2._shard_values[name] == np.array([2])).all()
