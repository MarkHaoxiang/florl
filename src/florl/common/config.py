from collections.abc import MutableMapping

from pydantic import BaseModel


class Config(BaseModel):
    @classmethod
    def from_raw(cls, config: MutableMapping):
        """
        Convert a dictionary to a Config object.
        """
        return cls(**config)
