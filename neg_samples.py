import json
import db_model
import numpy as np

def neg_entry2list(negSamples_entrys):
    returnList = []
    for entry in negSamples_entrys:  # 需要20毫秒
        returnList.append(
            {'id': entry[0], 'vector': np.array(json.loads(entry[2]))})
    return returnList


def get(contextWords, k=10):
    returnValues = db_model.getNegSameples(contextWords, k)

    uniqueSamples = []
    for entry in returnValues:
        
        if entry[1] not in contextWords:


            uniqueSamples.append(entry)
            if len(uniqueSamples) >= k:
                # 跳出循环
                break


    ls = neg_entry2list(uniqueSamples)
    return ls



