from fastapi import HTTPException, status

from ..constants import TARGETS
from ..db.crud import get_specified_sample

import motor.motor_asyncio


async def load_specified_sample(target: str, objectid: str , db: motor.motor_asyncio.AsyncIOMotorDatabase):
    

    if target not in TARGETS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f'Invalid target: {target}',
        )
    specified = await get_specified_sample(target, objectid , db)
    if specified is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f'No samples found',
        )
    return {
        'id': str(specified['_id']),
        'x': specified['x'],
        'y': specified['y'],
    }
