from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from core.db.utils import init_mongodb
from core.impl import (
    predict as predict_impl,
    load_sample as load_sample_impl,
    get_patients as get_patients_impl,
    get_patient as get_patient_impl,
    import_patient as import_patient_impl,
    import_ner as import_ner_impl
)
from core.models import PredictRequest
from core.utils import (
    init_grpc_client,
    init_norm_params
)

import os

vars = {}
app = FastAPI()

# configure CORS middleware if origins env var is set
cors_origins = '*'


if cors_origins is not None:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins.split(','),
        allow_credentials=True,
        allow_methods=['*'],
        allow_headers=['*'],
    )

# 启动后端


@app.on_event('startup')
def startup_event():
    vars['grpc_client'] = init_grpc_client()
    vars['norm_params'] = init_norm_params()
    vars['db_client'], vars['db'] = init_mongodb()

# 关闭后端


@app.on_event('shutdown')
def shutdown_event():
    if vars['db_client'] is not None:
        vars['db_client'].close()

# 以下前后端数据交互的方法定义，方法实现在implements里边
# 获取前端发送的数据并预测


@app.post('/api/predict')
async def predict(payload: PredictRequest):
    return predict_impl(
        payload=payload,
        grpc_client=vars['grpc_client'],
        norm_params=vars['norm_params'],
    )

# 向前端发送示例数据


@app.get('/api/load-sample')
async def load_sample(target: str):
    return await load_sample_impl(
        target=target,
        db=vars['db'],
    )

# 向前端发送病例数据


@app.get('/api/patients')
async def get_patients():
    return await get_patients_impl(db=vars['db'])

# 获取前端的病例数据


@app.post('/api/patients')
async def import_patient(
    id: str = Form(...),
    name: str = Form(...),
    age: int = Form(...),
    gender: str = Form(...),
    ethnicity: str = Form(...),
    importfile: UploadFile = File(...)
):
    print(name)
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

# 向前端发送某一特定id的病例数据


@app.get('/api/patients/{patient_id}')
async def get_patient(patient_id: str):
    return await get_patient_impl(
        patient_id=patient_id,
        db=vars['db'],
        grpc_client=vars['grpc_client'],
        norm_params=vars['norm_params'],
    )

# 获取前端的实体识别文本


@app.post('/api/ner')
async def import_ner(
    sequence: str = Form(...),
    # file_import: bool = Form(...),
    # importfile: UploadFile = File(...)
):
    print(sequence)
    return await import_ner_impl(
        db=vars['db'],
        sequence=sequence,
        # file_import=file_import,
        # importfile=importfile,
    )


#  TODO:
