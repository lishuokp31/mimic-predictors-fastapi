from pathlib import Path
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_log_pb2

import fire
import tensorflow as tf

TARGETS = {
    'aki': (1, 8, 16),
    'sepsis': (1, 14, 225),
    'mi': (1, 14, 221),
    'vancomycin': (1, 14, 224),
}


def create_warmup_records(saved_models_dir: str = 'saved_models'):
    saved_models_path = Path(saved_models_dir)

    for target, shape in TARGETS.items():
        model_path = saved_models_path / target / '1'

        # SavedModel warmup records are placed inside the "assets.extra" directory
        # so we need to create this directory if it doesn't exist yet
        extra_dir = model_path / 'assets.extra'
        extra_dir.mkdir(parents=False, exist_ok=True)

        # extract n_timesteps and n_features from target shape
        _, n_timesteps, n_features = shape

        # create a single inference sample to warmup the model
        # path is converted to string since TFRecordWriter doesn't support path
        records_path = extra_dir / 'tf_serving_warmup_requests'
        with tf.io.TFRecordWriter(str(records_path)) as writer:
            log = prediction_log_pb2.PredictionLog(
                predict_log=prediction_log_pb2.PredictLog(
                    request=predict_pb2.PredictRequest(
                        inputs={
                            'input_1': tf.make_tensor_proto(
                                [[[0] * n_features] * n_timesteps],
                                dtype=tf.float32,
                                shape=shape)
                        }
                    )
                )
            )
            writer.write(log.SerializeToString())


if __name__ == '__main__':
    fire.Fire(create_warmup_records)
