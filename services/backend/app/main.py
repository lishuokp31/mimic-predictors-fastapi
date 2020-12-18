from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from core.db.utils import init_mongodb
from core.impl import (
    predict as predict_impl,
    load_sample as load_sample_impl,
    get_patients as get_patients_impl,
    get_patient as get_patient_impl,
    import_patient as import_patient_impl,
)
from core.models import PredictRequest
from core.utils import (
    init_prediction_service_stub,
    init_norm_params
)

import os

vars = {}
app = FastAPI()

# configure CORS middleware if origins env var is set
# cors_origins = os.getenv('CORS_ORIGINS')
cors_origins = 'http://localhost:4200,http://192.168.1.3:4200'
if cors_origins is not None:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins.split(','),
        allow_credentials=True,
        allow_methods=['*'],
        allow_headers=['*'],
    )


@app.on_event('startup')
def startup_event():
    vars['stub'] = init_prediction_service_stub()
    vars['norm_params'] = init_norm_params()
    vars['client'], vars['db'] = init_mongodb()


@app.on_event('shutdown')
def shutdown_event():
    if vars['client'] is not None:
        vars['client'].close()


@app.post('/api/predict')
async def predict(payload: PredictRequest):
    return predict_impl(
        payload=payload,
        stub=vars['stub'],
        norm_params=vars['norm_params'],
    )


@app.get('/api/load-sample')
async def load_sample(target: str):
    return await load_sample_impl(
        target=target,
        db=vars['db'],
    )


@app.get('/api/patients')
async def get_patients():
    return await get_patients_impl(db=vars['db'])


@app.post('/api/patients')
async def import_patient(
    id: str = Form(...),
    name: str = Form(...),
    age: int = Form(...),
    gender: str = Form(...),
    ethnicity: str = Form(...),
    importfile: UploadFile = File(...)
):
    return await import_patient_impl(
        db=vars['db'],
        id=id,
        name=name,
        age=age,
        gender=gender,
        ethnicity=ethnicity,
        weight=0,
        height=0,
        import_file=importfile,
    )


@app.get('/api/patients/{patient_id}')
async def get_patient(patient_id: str):
    return await get_patient_impl(
        patient_id=patient_id,
        db=vars['db'],
        stub=vars['stub'],
        norm_params=vars['norm_params'],
    )
