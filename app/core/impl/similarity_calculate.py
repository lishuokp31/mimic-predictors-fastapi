import motor.motor_asyncio
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

record_limit_table = {
    # 'aki': 1452,
    'aki': 5,
    'mi': 5459,
    'sepsis': 5459,
    'vancomycin': 5459,
}


async def similarity_calculate(db: motor.motor_asyncio.AsyncIOMotorDatabase):

    target = 'aki'
    records = []

    record_limit = record_limit_table.get(target)
    
    async for record in get_all_record(target, db, record_limit):
        records.append(record)
    # print(currents)

    # similarities_table = np.zeros((record_limit, record_limit))
    similarities_title = []
    for record in records:
        similarities_title.append(str(record.get('_id')))
    similarities_table = []
    for i in range(0, record_limit):
        line = []
        for i in range(0, record_limit):
            line.append(0)

        similarities_table.append(line)
    # 获取当前需要比较的病历
    i = j = 0
    total_count = 1
    for current_record in records:
        print("============================================")
        print("current_record:" + current_record.get("_id").__str__())
        # 获取被比较的病历
        print("total_count:")
        print(total_count)
        total_count += 1
        total_count2 = 1
        for compared_record in records:
            # 避免一份病历与自己比较，以及两份病历被重复比较两次
            if (i != j) & (similarities_table[i][j] == 0) & (similarities_table[j][i] == 0):
                print("total_count2:")
                print(total_count2)
                total_count2 += 1
                print("compared_record:" + compared_record.get("_id").__str__())
                distance = get_distance(current_record, compared_record)
                similarities_table[i][j] = distance
                similarities_table[j][i] = distance
            # 与自己的相似度为1
            elif(i == j):
                similarities_table[j][i] = 1.0
            # 移动表下标
            if(j == record_limit - 1):
                j = 0
            else:
                j += 1
        # 移动表下标
        if(i == record_limit - 1):
            i = 0
        else:
            i += 1
    print("similarities_table:")
    print(similarities_table)
    result = await addSimilaritiestoDB(db, target, similarities_title, similarities_table)


async def get_all_record(target: str, db: motor.motor_asyncio.AsyncIOMotorDatabase, record_limit: int):
    query = {}
    projection = {'chartevents': 0}
    result = db[target].find(query).limit(record_limit)
    async for record in result:
        yield record


def get_days(current_record: any):
    x = current_record.get("x")
    days = 0
    for one_day in x:
        if(one_day[0] != 0):
            days += 1
    return days


def get_distance(current_record: any, compared_record: any):
    days1 = get_days(current_record)
    days2 = get_days(compared_record)
    x1 = current_record.get("x")
    print("x1:")
    print(x1)
    x2 = compared_record.get("x")
    print("x2:")
    print(x2)
    distances = []
    if(days1 == days2):
        print("x1 == x2")
        length = days1
        similarity_matrix = cosine_similarity(x1, x2)
        print(similarity_matrix)
        sum = 0
        for i in range(0, length):
            sum += similarity_matrix[i][i]
        distances.append(sum / length)

    elif(days1 >= days2):
        print("x1 is bigger")
        length = days2
        # print("days1:")
        # print(days1)
        # print("days2:")
        # print(days2)
        for i in range(0, days1 - days2 + 1):
            x1_tmp = x1[i:i + length]
            # print("x1_tmp:")
            # print(x1_tmp)
            similarity_matrix = cosine_similarity(x1_tmp, x2)
            print(similarity_matrix)
            sum = 0
            for i in range(0, length):
                sum += similarity_matrix[i][i]
            distances.append(sum / length)
    else:
        print("x2 is bigger")
        length = days1
        # print("days1:")
        # print(days1)
        # print("days2:")
        # print(days2)
        for i in range(0, days2 - days1 + 1):
            x2_tmp = x2[i:i + length]
            # print("x2_tmp:")
            # print(x2_tmp)
            similarity_matrix = cosine_similarity(x1, x2_tmp)
            print(similarity_matrix)
            sum = 0
            for i in range(0, length):
                sum += similarity_matrix[i][i]
            distances.append(sum / length)
    print("distances")
    print(distances)
    return max(distances)


async def addSimilaritiestoDB(db: motor.motor_asyncio.AsyncIOMotorDatabase, target: str, similarities_title: list, similarities_table: list):
    exist = await db.similarities.find_one({'target': target})
    if(exist is not None):
        modify_result = await db.similarities.update_one({'target': target}, {"$set": {"similarities_title": similarities_title, "similarities_table": similarities_table}})
    else:
        similarities_db_directory = {
            'target': target,
            "similarities_title": similarities_title,
            "similarities_table": similarities_table,
        }
        add_result = await db.similarities.insert_one(similarities_db_directory)
