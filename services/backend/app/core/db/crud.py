import motor.motor_asyncio


async def get_random_sample(target: str, db: motor.motor_asyncio.AsyncIOMotorDatabase):
    async for row in db[target].aggregate([{'$sample': {'size': 1}}]):
        return row

async def get_all_patients(db: motor.motor_asyncio.AsyncIOMotorDatabase, docs_per_page: int = 100):
    cursor = db.patients.find().limit(docs_per_page)
    async for document in cursor:
        yield document

async def get_one_patient(db: motor.motor_asyncio.AsyncIOMotorDatabase, patient_id: str):
    return await db.patients.find_one({'_id': patient_id})