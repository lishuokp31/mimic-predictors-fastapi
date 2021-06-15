import motor.motor_asyncio
import os


def init_mongodb():
    # get connection details from env
    hosts, port = os.environ['DB_HOSTS'], os.environ['DB_PORT']
    usr, pwd = os.environ['DB_USER'], os.environ['DB_PASSWORD']
    db, rs = os.environ['DB_DATABASE'], os.getenv('DB_REPLICA_SET')

    # `host` contain a comma-delimited hosts of the replica set nodes
    host = ','.join(f'{h}:{port}' for h in hosts.split(','))

    # build mongodb connection adding optional replica set if given
    uri = f'mongodb://{usr}:{pwd}@{host}/admin'
    if rs:
        uri += f'?replicaSet={rs}'
    # uri = 'mongodb://mongo-admin:passw0rd@host.docker.internal:27017/?authSource=admin'

    # initialize mongodb client
    client = motor.motor_asyncio.AsyncIOMotorClient(uri)

    # return specified database
    return client, client.get_database('mimic-predictors')
