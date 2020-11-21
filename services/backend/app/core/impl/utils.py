from typing import List
from ..models import ChartEvent

import pandas as pd
import typing


def parse_chart_events(csv_file: typing.IO) -> List[ChartEvent]:
    df = pd.read_csv(csv_file)
    df.columns = map(str.lower, df.columns)

    # the list of columns to retain
    wanted = ['charttime', 'itemid', 'value']

    # collect all of the events in the csv file
    # but only retaining the wanted columns
    events = df[wanted].values.tolist()
    events = map(lambda values: ChartEvent(*values), events)

    return events
