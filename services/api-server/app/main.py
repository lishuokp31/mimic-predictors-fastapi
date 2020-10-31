from fastapi import FastAPI

from core.db.utils import init_mongodb
from core.impl import (
    predict as predict_impl,
    load_sample as load_sample_impl,
)
from core.models import PredictRequest
from core.utils import init_prediction_service_stub, init_norm_params


app = FastAPI()
vars = {}


@app.on_event('startup')
def startup_event():
    vars['stub'] = init_prediction_service_stub()
    vars['norm_params'] = init_norm_params()
    vars['client'], vars['db'] = init_mongodb()


@app.on_event('shutdown')
def shutdown_event():
    # terminate all connections in the pool
    if vars['client'] is not None:
        vars['client'].close()


@app.post('/api/predict')
async def predict(payload: PredictRequest):
    stub, norm_params = vars['stub'], vars['norm_params']
    return await predict_impl(payload, stub, norm_params)


@app.get('/api/load-sample')
async def load_sample(target: str):
    db = vars['db']
    return await load_sample_impl(target, db)
