from pydantic import BaseModel
from typing import List


class PredictRequest(BaseModel):
    target: str
    x: List[List[int]]
