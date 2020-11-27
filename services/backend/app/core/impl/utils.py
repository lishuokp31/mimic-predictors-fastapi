import pandas as pd
import typing


def parse_chart_events(csv_file: typing.IO):
    df = pd.read_csv(csv_file)
    df.columns = map(str.lower, df.columns)

    # the list of columns to retain
    wanted = ['charttime', 'itemid', 'valuenum']

    # exclude rows with no values in the wanted fields
    df = df.dropna(axis=0, how='any', subset=wanted)

    # collect all of the events in the csv file
    # but only retaining the wanted columns
    return df[wanted].values.tolist()