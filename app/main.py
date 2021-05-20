from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from core.db.utils import init_mongodb
from core.impl import (
    predict as predict_impl,
    load_sample as load_sample_impl,
    load_specified_sample as load_specified_sample_impl,
    get_patients as get_patients_impl,
    get_patient as get_patient_impl,
    import_patient as import_patient_impl,
    import_ner as import_ner_impl,
    login as login_impl,
    register as register_impl,
    get_favorites as get_favorites_impl,
    # get_favorite as get_favorite_impl
    add_favorite as add_favorite_impl,
    delete_favorites as delete_favorites_impl,
    similarity_calculate as similarity_calculate_impl
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


@app.post('/api/load-specified-sample')
async def load_specified_sample(
    target: str = Form(...),
    objectid: str = Form(...),
):
    print("target:"+target)
    print("objectid:"+objectid)
    return await load_specified_sample_impl(
        target=target,
        objectid=objectid,
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


@app.post('/api/ner/txt')
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


@app.post('/api/ner/file')
async def import_ner2(
    file_sequence: str = Form(...)
    # file_import: bool = Form(...),
    # importfile: UploadFile = File(...)
):
    print("flagflagflagflagflagflagflagflagflagflagflagflagflag")
    for sequence in file_sequence:
        print(sequence)

    # return await import_ner_impl(
    #     db=vars['db'],
    #     sequence=sequence,
        # file_import=file_import,
        # importfile=importfile,
    # )
#  TODO:


@app.post('/api/login')
async def login(
    userName: str = Form(...),
    password: str = Form(...),
    # remember: bool = Form(...)
):
    print("userName:" + userName)
    print("password:" + password)
    return await login_impl(
        db=vars['db'],
        userName=userName,
        password=password,
        # remember=remember
    )


@app.post('/api/register')
async def register(
    username: str = Form(...),
    password: str = Form(...),
    email: str = Form(...),
    phone: str = Form(...),
):
    print("username:" + username)
    print("password:" + password)
    print("email:" + email)
    print("phone:" + phone)
    return await register_impl(
        db=vars['db'],
        username=username,
        password=password,
        email=email,
        phone=phone)


@app.get('/api/favorites/{username}')
async def get_favorites(username: str):
    return await get_favorites_impl(db=vars['db'] , username = username)

# @app.get('/api/favorites/{favorite_id}')
# async def get_favorite(favorite_id: str):
#     return await get_favorite_impl(
#         favorite_id=favorite_id,
#         db=vars['db'],
#         grpc_client=vars['grpc_client'],
#         norm_params=vars['norm_params'],
#     )

@app.post('/api/favorites/add')
async def add_favorite(
    username: str = Form(...),
    id: str = Form(...),
    fav_type: str = Form(...),
    remark: str = Form(...),
    value: str = Form(...),
):
    # print("username:" + username)
    # print("id:" + id)
    # print("fav_type:" + fav_type)
    # print("remark:" + remark)
    # print("value:" + value)
    return await add_favorite_impl(
        db=vars['db'],
        username=username, 
        id=id, 
        fav_type=fav_type, 
        remark=remark, 
        value=value
    )

@app.post('/api/favorites/delete')
async def delete_favorites(
    username: str = Form(...),
    id: str = Form(...),
):
    print('username' + username)
    print('id' + id)
    return await delete_favorites_impl(
        db=vars['db'],
        username = username, 
        id = id)

@app.post('/api/similarity_calculate')
async def similarity_calculate():
    await similarity_calculate_impl(
        db=vars['db'],
    )



# if (__name__ == "__main__"):
#     similarity_calculate() 