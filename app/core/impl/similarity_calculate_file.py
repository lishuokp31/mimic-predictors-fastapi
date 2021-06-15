import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
from operator import itemgetter

record_limit_table = {
    'aki': 1452,
    # 'aki': 5,
    # 'mi': 5459,
    # 'sepsis': 5459,
    # 'vancomycin': 5459,
    'mi': 1452,
    'sepsis': 1452,
    'vancomycin': 1452,
}


def similarity_calculate2():
    # target = 'aki'
    # target = 'mi'
    # target = 'sepsis'
    target = 'vancomycin'
    print("beign")
    records = []
    record_limit = record_limit_table.get(target)
    file_read = '../../../data/' + target + '-cut.json'
    with open(file_read, 'r') as f:
        records = json.load(f)
    # 取设定数量的记录
    print(len(records))
    records = records[0: record_limit]

    # 初始化表头
    similarities_table_title = []
    for record in records:
        similarities_table_title.append(str(record.get('_id').get('$oid')))
    # print(similarities_table_title)
    # 初始化表格
    similarities_table = []
    for i in range(0, record_limit):
        line = []
        for i in range(0, record_limit):
            line.append(0)
        similarities_table.append(line)
    # # print(similarities_table)

    # 行列计数器
    i = j = 0
    current_count = 1
    compared_count = 1

    # 获取当前需要比较的病历
    for current_record in records:
        # print("============================================")
        # print("current_record:" + current_record.get("_id").__str__())
        for compared_record in records:
            # 与自己的相似度为1
            if(i == j):
                similarities_table[j][i] = 1.0
            # 避免一份病历与自己比较，以及两份病历被重复比较两次
            elif (i != j) and (similarities_table[i][j] == 0) and (similarities_table[j][i] == 0):
                # print("compared_record:" + compared_record.get("_id").__str__())
                distance = get_distance(current_record, compared_record)
                similarities_table[i][j] = distance
                similarities_table[j][i] = distance
                # print("compared_count:" +str( compared_count))
                compared_count += 1
                
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
        print("current_count:" + str(current_count))
        current_count += 1
        compared_count = 1

    result = {
        "target": target,
        "similarities_table_title": similarities_table_title,
        "similarities_table": similarities_table,
    }

    file_write = '../../../data/' + target + '-similarities_table.json'
    with open(file_write, 'w') as f:
        json.dump(result, f)
    print("the end")


def get_distance(current_record: any, compared_record: any):
    # 获取有效天数
    days1 = get_days(current_record)
    days2 = get_days(compared_record)
    # print("days1:" + str(days1))
    # print("days2:" + str(days2))
    # 获取数据
    x1 = current_record.get("x")
    # print("x1:")
    # print(x1)
    x2 = compared_record.get("x")
    # print("x2:")
    # print(x2)
    distances = []
    # 有空病历
    if (days1 == 0 or days2 == 0) :
        if days1 != 0 and days2 == 0 :
            cosine_similarity_matrix = cosine_similarity(x1[0:days1], x2[0:days1],)
            # print(cosine_similarity_matrix)
            sum = 0
            # 取对角线上的有效数据
            for i in range(0, days1):
                sum += cosine_similarity_matrix[i][i]
            distances.append(sum / days1)
        elif days1 == 0 and days2 != 0 :
            cosine_similarity_matrix = cosine_similarity(x1[0:days2], x2[0:days2])
            # print(cosine_similarity_matrix)
            sum = 0
            # 取对角线上的有效数据
            for i in range(0, days2):
                sum += cosine_similarity_matrix[i][i]
            distances.append(sum / days2)
        else :
            distances.append(1)
    # 无空病历
    else:
        if(days1 == days2):
            # print("x1 == x2")
            length = days1
            cosine_similarity_matrix = cosine_similarity(x1, x2)
            # print(cosine_similarity_matrix)
            sum = 0
            # 取对角线上的有效数据
            for i in range(0, length):
                sum += cosine_similarity_matrix[i][i]
            distances.append(sum / length)
        elif(days1 >= days2):
            # print("x1 is bigger")
            length = days2

            for i in range(0, days1 - days2 + 1):
                x1_tmp = x1[i:i + length]
                # print("x1_tmp:")
                # print(x1_tmp)
                cosine_similarity_matrix = cosine_similarity(x1_tmp, x2)
                # print(cosine_similarity_matrix)
                sum = 0
                for i in range(0, length):
                    sum += cosine_similarity_matrix[i][i]
                distances.append(sum / length)
        else:
            # print("x2 is bigger")
            length = days1
            for i in range(0, days2 - days1 + 1):
                x2_tmp = x2[i:i + length]
                # print("x2_tmp:")
                # print(x2_tmp)
                cosine_similarity_matrix = cosine_similarity(x1, x2_tmp)
                # print(cosine_similarity_matrix)
                sum = 0
                for i in range(0, length):
                    sum += cosine_similarity_matrix[i][i]
                distances.append(sum / length)
    # print("distances")
    # print(distances)
    return max(distances)


def get_days(current_record: any):
    x = current_record.get("x")
    days = 0
    for one_day in x:
        if(one_day[0] != 0):
            days += 1
    return days

def file_handle():
    # target = 'aki'
    # target = 'mi'
    # target = 'sepsis'
    target = 'vancomycin'
    print("beign")
    records = []
    record_limit = record_limit_table.get(target)
    file_read = '../../../data/' + target + '.json'
    with open(file_read, 'r') as f:
        records = json.load(f)
    # 取设定数量的记录
    print(len(records))
    records = records[0: record_limit]

    file_write = '../../../data/' + target + '-cut.json'
    with open(file_write, 'w') as f:
        json.dump(records, f)
    print("the end")

def file_round():
    # target = 'aki'
    # target = 'mi'
    # target = 'sepsis'
    target = 'vancomycin'
    print("beign")
    records = []
    file_read = '../../../data/' + target + '-similarities_table.json'
    with open(file_read, 'r') as f:
        records = json.load(f)

    similarities_table_title = records[0].get('similarities_table_title')
    similarities_table = records[0].get('similarities_table')
    result = []
    length = len(similarities_table)
    for i in range(0 , length):
        similarities_record = {
            "target": target,
            "value" : similarities_table_title[i],
            "similarities_list":[]
        }
        similarities_list = []
        for j in range(0 , length):
            similarities_list.append({
                "similarity" : round(similarities_table[i][j] , 8),
                "value" : similarities_table_title[j],
            })
        similarities_list = sorted(similarities_list , key = itemgetter("similarity"), reverse=True)
        # print(similarities_list)
        similarities_record["similarities_list"] = similarities_list
        result.append(similarities_record)
        # break
    file_write = '../../../data/' + target + '-similarities_table_final.json'
    with open(file_write, 'w') as f:
        json.dump(result, f)
    print("the end")

def generate_array():
    array = []
    i = 0
    base = 0
    for i in range(0,31):
        array.append(base)
        base += 50
    with open("test.txt","w") as f:
        f.write(str(array))

if (__name__ == "__main__"):
    # similarity_calculate2()
    # file_handle()
    # file_round()
    generate_array()
    