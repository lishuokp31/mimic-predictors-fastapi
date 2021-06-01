import motor.motor_asyncio

async def get_users(
    db: motor.motor_asyncio.AsyncIOMotorDatabase,
):
    users = []
    async for user in get_all_users(db):
        del user['_id']
        users.append(user)
    return users

async def update_user(
    db: motor.motor_asyncio.AsyncIOMotorDatabase,
    username: str,
    modified_level : str
):
    modify_result = await db.users.update_one({"username": username}, {"$set": {"level": int(modified_level)}})

async def get_all_users(db: motor.motor_asyncio.AsyncIOMotorDatabase):
    records = db.users.find()
    async for user in records:
        yield user