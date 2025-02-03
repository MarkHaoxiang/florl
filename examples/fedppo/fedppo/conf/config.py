from pydantic import BaseModel


class Environment(BaseModel):
    env_id: str
