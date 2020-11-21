from ..models import Patient

import motor.motor_asyncio


async def get_random_sample(target: str, db: motor.motor_asyncio.AsyncIOMotorDatabase):
    async for row in db[target].aggregate([{'$sample': {'size': 1}}]):
        return row


async def get_all_patients(db: motor.motor_asyncio.AsyncIOMotorDatabase, docs_per_page: int = 100):
    query = {}
    projection = {'chartevents': 0}
    cursor = db.patients.find(query, projection).limit(docs_per_page)
    async for patient in cursor:
        yield patient


async def get_one_patient(db: motor.motor_asyncio.AsyncIOMotorDatabase, patient_id: str):
    query = {'_id': patient_id}
    projection = {'chartevents': 0}
    return await db.patients.find_one(query, projection)


async def insert_patient(db: motor.motor_asyncio.AsyncIOMotorDatabase, patient: Patient):
    # convert patient dataclass into a dictionary
    # also convert its chartevents list
    patient = patient._asdict()
    patient['chartevents'] = [
        event._asdict()
        for event in patient['chartevents']
    ]

    # convert patient id into an object id
    patient['_id'] = patient['id']
    del patient['id']

    # attempt to insert patient to database
    _ = await db.patients.insert_one(patient)
    return patient
