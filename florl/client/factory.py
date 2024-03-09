from abc import ABC, abstractmethod
from flwr.common import Config

from .client import FlorlClient
from florl.common import Knowledge


class FlorlFactory(ABC):
    """Constructor for reinforcement learning clients"""

    def __init__(self, config: Config) -> None:
        self._cfg = config

    @abstractmethod
    def create_default_knowledge(self, config: Config) -> Knowledge:
        raise NotImplementedError

    @abstractmethod
    def create_client(self, cid: str, config: Config) -> FlorlClient:
        raise NotImplementedError
