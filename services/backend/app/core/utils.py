from typing import Dict

from tensorflow_serving.apis import prediction_service_pb2_grpc
from tensorflow_serving.apis.prediction_service_pb2_grpc import PredictionServiceStub

import grpc
import numpy as np
import os
import pickle


def init_prediction_service_stub() -> PredictionServiceStub:
    host, port = os.environ['GRPC_HOST'], os.environ['GRPC_PORT']
    channel = grpc.aio.insecure_channel(f"{host}:{port}")
    return prediction_service_pb2_grpc.PredictionServiceStub(channel)


def init_norm_params() -> Dict[str, Dict[str, np.ndarray]]:
    with open('norm_params.p', 'rb') as fd:
        return pickle.load(fd)
