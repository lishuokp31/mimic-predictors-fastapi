from collections import namedtuple
from typing import List

from pydantic import BaseModel


class PredictRequest(BaseModel):
    target: str
    x: List[List[int]]


Patient = namedtuple('Patient', [
    'id', 'name', 'age', 'gender', 'weight',
    'height', 'ethnicity', 'added_at', 'updated_at', 'chartevents']
)
