from typing import Callable

import json
from flwr.common import Metrics as FlwrMetrics, MetricsAggregationFn

type JSONSerializable = (
    list[JSONSerializable] | dict[str, JSONSerializable] | str | int | float | bool
)

FLORL_RAW_METRICS = "florl_json"


class Metrics:
    def __init__(self, metrics: JSONSerializable):
        super().__init__()
        self._metrics = metrics

    def dump(self) -> FlwrMetrics:
        """
        Serialize the metrics to a JSON-encoded byte string and wrap it in a dictionary.

        Returns:
            fl.common.Metrics: A dictionary containing the JSON-encoded metrics.
        """
        raw = json.dumps(self._metrics).encode("utf-8")
        return {FLORL_RAW_METRICS: raw}

    @classmethod
    def load(cls, metrics: FlwrMetrics):
        """
        Deserialize the JSON-encoded metrics from a dictionary and create a Metrics instance.

        Args:
            metrics (fl.common.Metrics): A dictionary containing the JSON-encoded metrics.

        Raises:
            ValueError: If the key FLORL_RAW_METRICS is not found in the dictionary.

        Returns:
            Metrics: An instance of the Metrics class with the deserialized data.
        """
        if FLORL_RAW_METRICS not in metrics:
            raise ValueError(f"Key {FLORL_RAW_METRICS} not found in {metrics.keys()}")
        raw = metrics[FLORL_RAW_METRICS]
        assert isinstance(raw, bytes)
        return cls(json.loads(raw))

    @property
    def metrics(self) -> JSONSerializable:
        """
        Getter for the metrics attribute.

        Returns:
            JSONSerializable: The stored metrics.
        """
        return self._metrics


def transpose_dicts(
    list_of_dicts: list[dict[str, JSONSerializable]],
) -> dict[str, list[JSONSerializable]]:
    """
    Transpose a list of dictionaries into a dictionary of lists.

    Args:
        list_of_dicts (list[dict[str, JSONSerializable]]): List of dictionaries to transpose.

    Returns:
        dict[str, list[JSONSerializable]]: Dictionary with keys from input dictionaries and values as lists of corresponding values.
    """
    if len(list_of_dicts) == 0:
        return {}

    keys = set().union(*(d.keys() for d in list_of_dicts))
    transposed: dict[str, list[JSONSerializable]] = {key: [] for key in keys}

    for d in list_of_dicts:
        for d_key in d.keys():
            transposed[d_key].append(d[d_key])

    return transposed


def metrics_aggregation_fn(
    fn: Callable[[list[tuple[int, JSONSerializable]]], JSONSerializable],
) -> MetricsAggregationFn:
    """
    Create a metrics aggregation function.

    Args:
        fn (Callable[[list[tuple[int, JSONSerializable]]], JSONSerializable]):
            A function that takes a list of tuples, where each tuple contains an integer
            and JSON-serializable metrics, and returns aggregated JSON-serializable metrics.

    Returns:
        MetricsAggregationFn
    """

    def _fn(results: list[tuple[int, FlwrMetrics]]):
        return Metrics(
            fn([(n, Metrics.load(metrics).metrics) for n, metrics in results])
        ).dump()

    return _fn
