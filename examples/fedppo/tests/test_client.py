from flwr.common import Context, EvaluateIns, FitIns, GetParametersIns

from fedppo.client_app import client_fn


context = Context(0, 0, {}, {}, {})  # type: ignore


def test_create_client():
    client_fn(context)


def test_evaluate():
    client = client_fn(context)
    evaluate_res = client.evaluate(
        EvaluateIns(
            parameters=client.get_parameters(GetParametersIns({})).parameters,
            config={"max_steps": 10},
        )
    )

    assert "episode_reward" in evaluate_res.metrics


# def test_train():
#     client = client_fn(context)
#     client.fit(
#         FitIns(parameters=client.get_parameters(GetParametersIns({})).parameters),
#         config={},
#     )
