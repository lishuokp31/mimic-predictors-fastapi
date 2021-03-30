from collections import namedtuple
from typing import List

from pydantic import BaseModel


class PredictRequest(BaseModel):
    target: str
    x: List[List[float]]


Patient = namedtuple('Patient', [
    'id', 'name', 'age', 'gender', 'weight',
    'height', 'ethnicity', 'addedAt', 'updatedAt', 'chartevents']
)
