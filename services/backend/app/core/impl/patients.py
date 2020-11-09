from fastapi import HTTPException, status

from ..db.crud import (
    get_all_patients, 
    get_one_patient,
)

import motor.motor_asyncio

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

    return patient