from datetime import datetime
from fastapi import HTTPException, status, UploadFile
from min_tfs_client.requests import TensorServingClient
from typing import Dict
import motor.motor_asyncio

from ..constants import GENDERS, ETHNICITIES
from ..db.crud import (
    insert_ner,
)
from ..models import Ner
from .utils import (
    parse_chartevents,
    reduce_chartevents,
    extract_metrics,
    extract_disease_probabilities,
)
from ..ner.classify import do_classify

import numpy as np


async def import_ner(
    db: motor.motor_asyncio.AsyncIOMotorDatabase,
    sequence: str,
    # file_import: bool,
    # importfile: UploadFile,
):
    # 文本输入
    # if not file_import :
    entities = do_classify(sequence)

    await insert_ner(db, entities, sequence)

    entities_directory = []
    # entities_model = ["kind", "start_index", "end_index", "actual_value"]
    for entity in entities:
        # entities_directory.append(dict(zip(entities_model, entity)))
        entities_directory.append(entity)

    ner_db_directory = {
        # "_id": "id", TODO:
        "sequence": sequence,
        "entities": entities_directory
    }

    print(ner_db_directory)
    return ner_db_directory
