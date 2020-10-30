from fastapi import FastAPI

from core.models import PredictRequest
from core.impl import predict as predict_impl


app = FastAPI()


@app.post('/api/predict')
async def predict(payload: PredictRequest):
    return await predict_impl(payload)


@app.get('/api/load-sample')
async def load_sample(target: str):
    pass
