from flwr.common import FitRes, EvaluateRes, Status, Code, Metrics, Parameters


def fit_ok(
    num_examples: int,
    parameters: Parameters,
    metrics: Metrics,
    message: str = "Success",
):
    return FitRes(
        status=Status(Code.OK, message),
        num_examples=num_examples,
        parameters=parameters,
        metrics=metrics,
    )


def evaluate_ok(num_examples: int, metrics: Metrics, message: str = "Success"):
    return EvaluateRes(
        status=Status(Code.OK, message),
        loss=0.0,
        num_examples=num_examples,
        metrics=metrics,
    )
