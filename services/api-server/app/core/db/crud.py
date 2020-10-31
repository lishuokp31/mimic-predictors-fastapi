import motor.motor_asyncio


async def get_random_sample(target: str, db: motor.motor_asyncio.AsyncIOMotorDatabase):
    async for row in db[target].aggregate([{'$sample': {'size': 1}}]):
        return row
