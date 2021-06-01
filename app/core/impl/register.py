import motor.motor_asyncio


async def register(
    db: motor.motor_asyncio.AsyncIOMotorDatabase,
    username: str,
    password: str,
    email: str,
    phone: str,
):
    result = await db.users.find_one({"username": username})
    print(result)
    if(result is not None):
        print("该用户名已存在")
        return 0
    else:
        register_db_directory = {
            "username": username,
            "password": password,
            "email": email,
            "phone": phone,
            "level": -1,
        }

        _ = await db.users.insert_one(register_db_directory)
        result2 = await db.users.find_one({"username": username})
        print(result2)
        if(result2 is not None):
            print("注册成功！")
            favorite_list = []
            favorite_db_directory = {
                "username": username,
                "favorite_list": favorite_list,
            }
            add_result = await db.favorites.insert_one(favorite_db_directory)
            return 1
        print("注册失败，请联系管理员解决")
        return -1
