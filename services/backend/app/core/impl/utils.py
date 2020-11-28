from tensorflow_serving.apis.prediction_service_pb2_grpc import PredictionServiceStub
from typing import Any, Dict, List

from .predict import predict
from ..constants import (
    ITEM_IDS, MI_FEATURES, REV_FEATURE_MAP, N_TIMESTEPS,
    AKI_FEATURES, SEPSIS_FEATURES, VANCOMYCIN_FEATURES,
)
from ..models import PredictRequest

import numpy as np
import pandas as pd
import typing

# the list of columns to retain
WANTED = ['charttime', 'itemid', 'valuenum']


def parse_chartevents(csv_file: typing.IO):
    df = pd.read_csv(csv_file)
    df.columns = map(str.lower, df.columns)

    # exclude rows with no values in the wanted fields
    df = df.dropna(axis=0, how='any', subset=WANTED)

    # collect all of the events in the csv file
    # but only retaining the wanted columns
    return df[WANTED].values.tolist()


def reduce_chartevents(chartevents):
    df = pd.DataFrame(chartevents, columns=WANTED)

    # preprocessing steps
    # 1. remove rows with nan values on the wanted columns
    # 2. remove rows with item IDs not on the feature list
    condition = df['itemid'].isin(ITEM_IDS)
    df = df[condition].dropna(axis=0, how='any', subset=WANTED)

    # unit conversions
    # weight with itemid = 226707 and 1394 are in inches (convert to cm)
    mask = (df['itemid'] == 226707) | (df['itemid'] == 1394)
    df.loc[mask, 'valuenum'] *= 2.54

    # group feature values by days and features
    df['chartday'] = df['charttime'].astype(
        'str').str.split(' ').apply(lambda x: x[0])
    df['feature'] = df['itemid'].apply(lambda x: REV_FEATURE_MAP[x])

    # get mean, max, min, std of feature values for each day
    df_mean = pd.pivot_table(
        df, index='chartday', columns='feature', values='valuenum',
        fill_value=np.nan, aggfunc=np.nanmean, dropna=False,
    )
    df_max = pd.pivot_table(
        df, index='chartday', columns='feature', values='valuenum',
        fill_value=np.nan, aggfunc=np.nanmax, dropna=False,
    )
    df_max.columns = [f'{c}_max' for c in df_mean.columns]
    df_min = pd.pivot_table(
        df, index='chartday', columns='feature', values='valuenum',
        fill_value=np.nan, aggfunc=np.nanmin, dropna=False,
    )
    df_min.columns = [f'{c}_min' for c in df_mean.columns]
    df_std = pd.pivot_table(
        df, index='chartday', columns='feature', values='valuenum',
        fill_value=np.nan, aggfunc=np.nanstd, dropna=False,
    )
    df_std.columns = [f'{c}_std' for c in df_mean.columns]

    return pd.concat([df_mean, df_max, df_min, df_std], axis=1)


def extract_metrics(chartevents: pd.DataFrame):
    return [
        {
            'label': '体温',
            'unit': '℃',
            'mean': chartevents['temperature'].iloc[-1].item(),
            'min': chartevents['temperature_min'].iloc[-1].item(),
            'max': chartevents['temperature_max'].iloc[-1].item(),
            'std': chartevents['temperature_std'].iloc[-1].item(),
        },
        {
            'label': '心率',
            'unit': 'bpm',
            'mean': chartevents['heart rate'].iloc[-1].item(),
            'min': chartevents['heart rate_min'].iloc[-1].item(),
            'max': chartevents['heart rate_max'].iloc[-1].item(),
            'std': chartevents['heart rate_std'].iloc[-1].item(),
        },
        {
            'label': '收缩压',
            'unit': 'mmHg',
            'mean': chartevents['systolic'].iloc[-1].item(),
            'min':  chartevents['systolic_min'].iloc[-1].item(),
            'max':  chartevents['systolic_max'].iloc[-1].item(),
            'std':  chartevents['systolic_std'].iloc[-1].item(),
        },
        {
            'label': '舒张压',
            'unit': 'mmHg',
            'mean': chartevents['diastolic'].iloc[-1].item(),
            'min': chartevents['diastolic_min'].iloc[-1].item(),
            'max': chartevents['diastolic_max'].iloc[-1].item(),
            'std': chartevents['diastolic_std'].iloc[-1].item(),
        }
    ]


def extract_disease_probabilities(
    patient: Dict[str, Any],
    df: pd.DataFrame,
    stub: PredictionServiceStub,
    params: Dict[str, Dict[str, np.ndarray]],
):
    # extract static information from patient
    df['gender'] = df['M'] = 1 if patient['gender'] == '男' else 0
    df['age'] = patient['age']
    df['black'] = df['BLACK'] = 0

    # do forward/backward imputation to fill holes
    df = df.ffill().bfill()

    # for patients with long stays (n_days > 14)
    # we only take the recent 14 days into account
    df = df.iloc[-N_TIMESTEPS['sepsis']:]

    return {
        'aki': do_predict(df.copy(), stub, params, 'aki', AKI_FEATURES),
        'sepsis': do_predict(df.copy(), stub, params, 'sepsis', SEPSIS_FEATURES),
        'mi': do_predict(df.copy(), stub, params, 'mi', MI_FEATURES),
        'vancomycin': do_predict(df.copy(), stub, params, 'vancomycin', VANCOMYCIN_FEATURES),
    }


def do_predict(
    df: pd.DataFrame,
    stub: PredictionServiceStub,
    norm_params: Dict[str, Dict[str, np.ndarray]],
    target: str,
    features: List[str],
):
    # further trim the dataframe to only contain the
    # recent 8 days if target is AKI
    if target == 'aki':
        df = df.iloc[-N_TIMESTEPS['aki']:]

    try:
        x = df[features].values.tolist()
        payload = PredictRequest(target=target, x=x)
        response = predict(payload, stub, norm_params)
        missing_features = []
        predictions, weights = response['predictions'], response['weights']
    except KeyError as e:
        # part of the message exception are the feature names which doesn't
        # contain any values, so we parse it to get its value
        e_str = str(e)
        start, end = e_str.index('[') + 1, e_str.index(']')
        missing_features = [mf[1:-1] for mf in e_str[start:end].split(', ')]

        # we fill missing values with nan so that the user will know which
        # values are missing and should be added (if needed)
        df[missing_features] = np.nan
        x = df[features].values.tolist()
        predictions, weights = None, None

    # transform nans into None type
    x = [[None if np.isnan(__x) else __x for __x in _x] for _x in x]

    return {
        'x': x,
        'weights': weights,
        'predictions': predictions,
        'nMissingFeatures': len(missing_features),
    }
