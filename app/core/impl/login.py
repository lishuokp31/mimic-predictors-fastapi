import motor.motor_asyncio


async def login(
    db: motor.motor_asyncio.AsyncIOMotorDatabase,
    userName: str, 
    password: str, 
    remember: bool
):
    result = await db.users.find_one({"username" : userName})
    print(result)
    print("-----------------------------------------")
    if(result.get('password') != password) : 
        return {
            "login" : False,
        }
    return {
        "login" : True,
        "username" : result.get('username'),
        "email" : result.get('email'),
        "phone" : result.get('phone'),
        "level" : result.get('level'),
    }