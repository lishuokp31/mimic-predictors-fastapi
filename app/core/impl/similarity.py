import motor.motor_asyncio
import numpy as np

async def similarity(db: motor.motor_asyncio.AsyncIOMotorDatabase , current_id :str):
    record = await db.similarities.find_one({"value" : current_id})
    similarities_list =record.get("similarities_list")
    result = []
    for i in range(1, 21):
        result.append({
            "id": "{:0>3d}".format(i),
            "similarity" : similarities_list[i].get("similarity"),
            "value" : similarities_list[i].get("value"),
            })
    return result
