from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
from typing import List

from tensorflow import make_tensor_proto, TensorShape
from tensorflow.core.framework import types_pb2
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc

import grpc
import numpy as np
import os
import pickle

"""
TODO: add comment
"""
TARGETS = ['aki', 'sepsis', 'mi', 'vancomycin']
N_FEATURES = {
    'sepsis': 225,
    'mi': 221,
    'vancomycin': 224,
    'aki': 16,
}
N_TIMESTEPS = {
    'sepsis': 14,
    'mi': 14,
    'vancomycin': 14,
    'aki': 8,
}

"""
TODO: add comment
"""


class PredictRequest(BaseModel):
    target: str
    x: List[List[int]]


"""
TODO: add comment
"""
app = FastAPI()

"""
TODO: add comment
"""
host, port = os.environ['GRPC_HOST'], os.environ['GRPC_PORT']
channel = grpc.aio.insecure_channel(f"{host}:{port}")
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

"""
TODO: add comment
"""
with open('norm_params.p', 'rb') as fd:
    norm_params = pickle.load(fd)


def preprocess(payload: PredictRequest):
    # target refers to the model the caller wants to use
    target = payload.target

    # target should be one of the supported targets
    if target not in TARGETS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f'Invalid target "{target}"',
        )

    # retrieve input data x from request body
    x = payload.x

    # attempt to convert input data into a tensor
    # conversion to tensor fails if x contains non-numeric values
    try:
        x = np.array(x, dtype=np.float32)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail='Input data contains non-numeric values',
        )

    # input matrix x should be a 2-dimensional tensor
    if len(x.shape) != 2:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail='Input data should be a 2-dimensional tensor',
        )

    # extract timesteps and no. of features from the matrix shape
    n_timesteps, n_features = x.shape

    # check timesteps (maximum of 14 days for sepsis/mi/vancomycin)
    # and maximum of 8 days for AKI
    if n_timesteps > N_TIMESTEPS[target]:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f'Target="{target}" maximum timesteps is only 8',
        )

    # check the input matrix's number of features
    if n_features != N_FEATURES[target]:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f'Target="{target}" no. of features should be {N_FEATURES[target]}',
        )

    # we allow frontends to pass lesser timesteps
    # so we need to fill these with zeroes
    timesteps_diff = N_TIMESTEPS[target] - n_timesteps
    if timesteps_diff > 0:
        extra = np.zeros((timesteps_diff, n_features))
        x = np.concatenate([x, extra], axis=0)

    # add batch dimension to input matrix
    x = np.expand_dims(x, axis=0)

    # manual input normalization is needed for legacy models (except AKI)
    if target != 'aki':
        x = normalize(norm_params[target], x)

    return x, target


def create_grpc_request(x, target):
    return predict_pb2.PredictRequest(
        model_spec={'name': target},
        inputs={'input_1': make_tensor_proto(
            x, dtype=types_pb2.DT_FLOAT, shape=(1, 8, 16)
        )}
    )


def postprocess(response):
    # extract predictions and weights from the proto response object
    predictions = response.outputs['output_1']
    weights = response.outputs['output_2']
    _, n_days, n_features = TensorShape(weights.tensor_shape)

    # this needs to be converted into a numpy array first (for convenience)
    # also, protobuf always flattens tensor so we need to reshape it back to its original shape
    # finally, we convert them from tensors to python list (for JSON serialization)
    predictions = np.array(predictions.float_val).tolist()
    weights = np.array(weights.float_val).reshape(n_days, n_features).tolist()

    # pack outputs as a python dict
    # to be serialized and passed as a json response
    return {
        'predictions': predictions,
        'weights': weights
    }


def normalize(params, x):
    # mask padding values with nan to not include it in normalization
    bool_matrix = ~x.any(axis=2)
    x[bool_matrix] = np.nan

    # normalize data
    x -= params['means']
    x /= params['stds']

    # replace NaNs with padding_value=0
    bool_matrix = np.isnan(x)
    x[bool_matrix] = 0

    return x


@app.post('/api/predict')
async def predict(payload: PredictRequest):
    x, target = preprocess(payload)
    request = create_grpc_request(x, target)
    response = await stub.Predict(request)
    return postprocess(response)
