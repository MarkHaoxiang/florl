from flwr.common import Context
from fedppo.client_app import client_fn


def test_create_client():
    client_fn(Context(0, 0, {}, {}, {}))
