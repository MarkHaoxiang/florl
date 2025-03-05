import os

import pytest
from omegaconf import OmegaConf
from flwr.common import Context, EvaluateIns, GetParametersIns
from florl.common import Metrics

from fedppo.client_app import client_fn
from fedppo.main import FedPPOConfig, CONFIG_DIR


def get_config_files():
    return [
        os.path.join(CONFIG_DIR, f)
        for f in os.listdir(CONFIG_DIR)
        if f.endswith(".yaml")
    ]


@pytest.fixture(scope="module", params=get_config_files())
def config(request) -> FedPPOConfig:
    with open(request.param, "r") as f:
        return OmegaConf.to_object(OmegaConf.load(f))  # type: ignore


@pytest.fixture
def context():
    return Context(0, 0, {}, {}, {})  # type: ignore


def test_create_client(config, context):
    client_fn(context, config["task"], config["client"])


def test_evaluate(config, context):
    client = client_fn(context, config["task"], config["client"]).client
    evaluate_res = client.evaluate(
        EvaluateIns(
            parameters=client.get_parameters(GetParametersIns({})).parameters,
            config={"max_steps": 10},
        )
    )
    metrics = Metrics.load(evaluate_res.metrics).metrics

    assert "episode_reward" in metrics
