import motor.motor_asyncio
import os


def init_mongodb():
    # get connection details from env
    host, port = os.environ['DB_HOST'], os.environ['DB_PORT']
    user, password = os.environ['DB_USER'], os.environ['DB_PASSWORD']
    database = os.environ['DB_DATABASE']

    # initialize mongodb client
    uri = f'mongodb://{user}:{password}@{host}:{port}'
    client = motor.motor_asyncio.AsyncIOMotorClient(uri)

    # return specified database
    return client, client[database]
