from fastapi import HTTPException, status

from ..constants import TARGETS
from ..db.crud import get_random_sample

import motor.motor_asyncio


async def load_sample(target: str, db: motor.motor_asyncio.AsyncIOMotorDatabase):
    # verify target is valid
    if target not in TARGETS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f'Invalid target: {target}',
        )

    # retrieve random sample from the target table
    sample = await get_random_sample(target, db)

    # there are times when the db is not initialized well
    # then no sample can be returned
    if sample is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'No samples found',
        )

    # format random sample
    return {
        'id': str(sample['_id']),
        'x': sample['x'],
        'y': sample['y'],
    }
