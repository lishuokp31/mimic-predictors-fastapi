from typing import Dict
from fastapi import HTTPException, status

from min_tfs_client.requests import TensorServingClient
from min_tfs_client.tensors import tensor_proto_to_ndarray

from ..models import PredictRequest
from ..constants import MODEL_ARCHITECTURES, TARGETS, N_FEATURES, N_TIMESTEPS

import numpy as np


def preprocess(
    payload: PredictRequest,
    norm_params: Dict[str, Dict[str, np.ndarray]],
):
    # the model architecture to be used for the current inference request
    architecture = payload.architecture
    if architecture not in MODEL_ARCHITECTURES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f'Invalid model architecture "{architecture}"',
        )

    # target refers to the model the caller wants to use
    target = payload.target
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

    return architecture, target, x


def postprocess(response):
    # extract predictions and weights from the proto response object
    predictions = response.outputs['output_1']
    weights = response.outputs['output_2']

    # this needs to be converted into a numpy array first (for convenience)
    # finally, we convert them from tensors to python list (for JSON serialization)
    predictions = tensor_proto_to_ndarray(predictions).reshape(-1).tolist()
    weights = tensor_proto_to_ndarray(weights).squeeze().tolist()

    # pack outputs as a python dict
    # to be serialized and passed as a json response
    return {
        'predictions': predictions,
        'weights': weights
    }


def normalize(params, x):
    # since the masking done is axis sensitive, we need to
    # make sure that x is of the correct shape [batch, n_days, n_features]
    assert x.ndim == 3

    # only normalize days with real data (as opposed to padding data)
    # days with only padding values will not be normalized (they're just 0s).
    # also, standard deviation values should not contain zeroes
    # so we add a small LayerNorm-esque epsilon value.
    mask = x.any(axis=-1)
    x[mask] -= params['means']
    x[mask] /= params['stds'] + 1e-5

    return x


def predict(
    payload: PredictRequest,
    grpc_client: TensorServingClient,
    norm_params: Dict[str, Dict[str, np.ndarray]],
):
    # validate and clean request payload
    archi, target, x = preprocess(payload, norm_params)

    # pass data to TensorFlow serving for inference
    response = grpc_client.predict_request(
        model_name=target,
        model_version=1,
        input_dict={
            'input_1': x.astype(np.float32)
        }
    )

    # format and return prediction results
    return postprocess(response)
