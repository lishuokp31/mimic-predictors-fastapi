from datetime import datetime
from fastapi import HTTPException, status, UploadFile

from ..constants import GENDERS, ETHNICITIES
from ..db.crud import (
    get_all_patients,
    get_one_patient,
    insert_patient,
)
from ..models import Patient
from .utils import parse_chart_events

import motor.motor_asyncio
import random


async def get_patients(db: motor.motor_asyncio.AsyncIOMotorDatabase):
    # we transform patients cursor into a list by ourselves
    # since we want to map the IDs to become JSON serializable
    patients = []

    async for patient in get_all_patients(db, docs_per_page=100):
        # we would want all of the values in the document except for the '_id' property
        # so we change it here
        patient['id'] = str(patient['_id'])
        del patient['_id']

        # add patient as-is
        patients.append(patient)

    return patients


async def get_patient(db: motor.motor_asyncio.AsyncIOMotorDatabase, patient_id: str):
    patient = await get_one_patient(db, patient_id)

    if patient is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f'Patient not found',
        )

    # we replace the `_id` property with a simple `id` property of type string
    patient['id'] = str(patient['_id'])
    del patient['_id']

    # add key patient metrics and disease probabilities
    # TODO: change to reflect data in database
    patient['metrics'] = [
        {
            'label': '体温',
            'unit': '℃',
            'mean': random.normalvariate(36.2, 0.63),
            'min': random.normalvariate(36.2, 0.63),
            'max': random.normalvariate(36.2, 0.63),
            'std': 0.63,
        },
        {
            'label': '心率',
            'unit': 'bpm',
            'mean': random.normalvariate(63, 4.19),
            'min': random.normalvariate(63, 4.19),
            'max': random.normalvariate(63, 4.19),
            'std': 4.19,
        },
        {
            'label': '收缩压',
            'unit': 'mmHg',
            'mean': random.normalvariate(120, 7.36),
            'min': random.normalvariate(120, 7.36),
            'max': random.normalvariate(120, 7.36),
            'std': 7.36,
        },
        {
            'label': '舒张压',
            'unit': 'mmHg',
            'mean': random.normalvariate(80, 6.85),
            'min': random.normalvariate(80, 6.85),
            'max': random.normalvariate(80, 6.85),
            'std': 6.85,
        }
    ]
    patient['probabilities'] = {
        'aki': [random.uniform(0, 1) for _ in range(8)],
        'sepsis': [random.uniform(0, 1) for _ in range(14)],
        'mi': [random.uniform(0, 1) for _ in range(14)],
        'vancomycin': [random.uniform(0, 1) for _ in range(14)],
    }

    return patient


async def import_patient(
    db: motor.motor_asyncio.AsyncIOMotorDatabase,
    id: str,
    name: str,
    age: int,
    gender: str,
    ethnicity: str,
    weight: float,
    height: float,
    import_file: UploadFile,
):
    # validate ID
    if len(id) < 1:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f'Invalid ID: "{id}"',
        )

    # validate name
    if len(name) < 1:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f'Invalid name: "{name}"',
        )

    # validate age
    if not 0 < age < 200:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f'Invalid age: {age}',
        )

    # validate gender
    if gender not in GENDERS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f'Invalid gender: "{gender}"',
        )

    # validate ethnicity
    if ethnicity not in ETHNICITIES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f'Invalid ethnicity: "{ethnicity}"',
        )

    # attempt to parse the uploaded CSV file
    try:
        chartevents = parse_chart_events(import_file.file)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )

    # attempt to insert the patient to the database
    patient = await insert_patient(db, Patient(
        id=id, name=name, age=age,
        weight=weight, height=height,
        gender=gender, ethnicity=ethnicity,
        added_at=datetime.today(), updated_at=datetime.today(),
        chartevents=chartevents,
    ))

    # we replace the `_id` property with a simple `id` property of type string
    patient['id'] = str(patient['_id'])
    del patient['_id']

    return patient
