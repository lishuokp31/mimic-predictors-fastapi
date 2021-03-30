import motor.motor_asyncio
import os


def init_mongodb():
    # # get connection details from env
    # hosts, port = os.environ['DB_HOSTS'], os.environ['DB_PORT']
    # usr, pwd = os.environ['DB_USER'], os.environ['DB_PASSWORD']
    # db, rs = os.environ['DB_DATABASE'], os.environ['DB_REPLICA_SET']

    # # `host` contain a comma-delimited hosts of the replica set nodes
    # host = ','.join(f'{h}:{port}' for h in hosts.split(','))

    # initialize mongodb client
    # uri = f'mongodb://{usr}:{pwd}@{host}/{db}?authSource={db}&replicaSet={rs}'
    uri = 'mongodb://mongo-admin:passw0rd@localhost:27017/?authSource=admin'
    client = motor.motor_asyncio.AsyncIOMotorClient(uri)

    # return specified database
    return client, client.get_database('mimic-predictors')
