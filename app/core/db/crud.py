from ..models import Patient , Ner
from bson import ObjectId
import motor.motor_asyncio


async def get_random_sample(target: str, db: motor.motor_asyncio.AsyncIOMotorDatabase):
    async for row in db[target].aggregate([{'$sample': {'size': 1}}]):
        print(row)
        return row

async def get_specified_sample(target: str, objectid : str , db: motor.motor_asyncio.AsyncIOMotorDatabase):
    result = await db[target].find_one({"_id" : ObjectId(objectid)})
    print(result)
    return result


async def get_all_patients(db: motor.motor_asyncio.AsyncIOMotorDatabase, docs_per_page: int = 100):
    query = {}
    projection = {'chartevents': 0}
    cursor = db.patients.find(query, projection).limit(docs_per_page)
    async for patient in cursor:
        yield patient


async def get_one_patient(db: motor.motor_asyncio.AsyncIOMotorDatabase, patient_id: str):
    query = {'_id': patient_id}
    return await db.patients.find_one(query)


async def insert_patient(db: motor.motor_asyncio.AsyncIOMotorDatabase, patient: Patient):
    # convert patient dataclass into a dictionary
    patient = patient._asdict()

    # convert patient id into an object id
    patient['_id'] = patient['id']
    del patient['id']

    # attempt to insert patient to database
    _ = await db.patients.insert_one(patient)
    return patient

async def insert_ner(db: motor.motor_asyncio.AsyncIOMotorDatabase, entities: list, sequence: str,):
    # ret = db.counters.findAndModify({"_id": 'productid'}, {"$inc": {"sequence_value": 1}}, safe=True, new=True)

    # nextid = ret["sequence_value"]
    
    entities_directory = []
    # entities_model = ["kind", "start_index", "end_index", "actual_value"]
    for entity in entities:
        # entities_directory.append(dict(zip(entities_model, entity)))
        entities_directory.append(entity)

    ner_db_directory = {
        # "_id": "id", TODO:
        "sequence": sequence,
        "entities": entities_directory
    }
    
    # ner_db_directory = ner._asdict()
    _ = await db.ner.insert_one(ner_db_directory)

    return ner_db_directory