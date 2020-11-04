from typing import Dict

from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow_serving.apis.prediction_service_pb2_grpc import PredictionServiceStub

from .interceptors import RetryOnRpcErrorClientInterceptor, ExponentialBackoff

import grpc
import numpy as np
import os
import pickle


def init_prediction_service_stub() -> PredictionServiceStub:
    # initialize retry interceptor
    interceptors = (
        RetryOnRpcErrorClientInterceptor(
            max_attempts=4,
            sleeping_policy=ExponentialBackoff(
                init_backoff_ms=100,
                max_backoff_ms=1600,
                multiplier=2,
            ),
            status_for_retry=(
                grpc.StatusCode.UNAVAILABLE,
            ),
        ),
    )

    # initialize channel and stub
    host, port = os.environ['GRPC_HOST'], os.environ['GRPC_PORT']
    channel = grpc.intercept_channel(
        grpc.insecure_channel(f"{host}:{port}"),
        *interceptors,
    )
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    return stub


def init_norm_params() -> Dict[str, Dict[str, np.ndarray]]:
    with open('norm_params.p', 'rb') as fd:
        return pickle.load(fd)
