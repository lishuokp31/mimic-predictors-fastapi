from typing import Dict
from min_tfs_client.requests import TensorServingClient

import numpy as np
# import os
import pickle

# 初始化grpc服务，设置主机与端口
def init_grpc_client() -> TensorServingClient:
    # host, port = os.environ['GRPC_HOST'], os.environ['GRPC_PORT']
    host, port = 'host.docker.internal', 9090
    return TensorServingClient(host=host, port=port, credentials=None)

# 初始化参数
def init_norm_params() -> Dict[str, Dict[str, np.ndarray]]:
    with open('norm_params.p', 'rb') as fd:
        return pickle.load(fd)
