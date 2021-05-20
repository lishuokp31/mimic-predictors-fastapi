import motor.motor_asyncio
import numpy as np


async def get_favorites(
    db: motor.motor_asyncio.AsyncIOMotorDatabase, username: str
):
    favorites = await db.favorites.find_one({'username': username})
    print("--------------------favorites--------------------")
    print(favorites)
    ("--------------------favorites end--------------------")
    result = []
    for favorite_entity in favorites.get('favorite_list'):
        result.append({
            'id': favorite_entity[0],
            'fav_type': favorite_entity[1],
            'remark': favorite_entity[2],
            'value': favorite_entity[3],
        })

    return result


# async def get_favorite():


async def add_favorite(
    db: motor.motor_asyncio.AsyncIOMotorDatabase,
    username: str,
    id: str,
    fav_type: str,
    remark: str,
    value: str
):
    user_fav = await db.favorites.find_one({"username": username})
    if(user_fav is not None):
        favorite_list = user_fav.get('favorite_list')
        add_line = [id, fav_type, remark, value]
        favorite_list.append(add_line)
        print("favorite_list:")
        print(favorite_list)
        modify_result = await db.favorites.update_one({"username": username}, {"$set": {"favorite_list": favorite_list}})
        ("--------------------modify_result--------------------")
        print(modify_result)
        ("--------------------modify_result end--------------------")
    else:
        favorite_list = [[id, fav_type, remark, value]]
        favorite_db_directory = {
            "username": username,
            "favorite_list": favorite_list,
        }
        add_result = await db.favorites.insert_one(favorite_db_directory)
        ("--------------------add_result--------------------")
        print(add_result)
        ("--------------------add_result end--------------------")


async def delete_favorites(
    db: motor.motor_asyncio.AsyncIOMotorDatabase, username: str,
    id: str,
):
    print("username:")
    print(username)
    print("id:")
    print(id)
    user_fav = await db.favorites.find_one({"username": username})
    favorite_list = user_fav.get('favorite_list')
    i = 0
    for fav in favorite_list:
        if fav[0] == id:
            del favorite_list[i]
        i += 1
    print("favorite_list:")
    print(favorite_list)
    modify_result = await db.favorites.update_one({"username": username}, {"$set": {"favorite_list": favorite_list}})

