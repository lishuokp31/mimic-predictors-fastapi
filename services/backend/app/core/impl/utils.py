from ..constants import ITEM_IDS, REV_FEATURE_MAP

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


def extract_metrics(chartevents):
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
