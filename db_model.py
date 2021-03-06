import mysql.connector
import json
import numpy as np

def connect2Mysql():
    conn = mysql.connector.connect(
        user='root', password='as56210', database='flow', use_unicode=True)
    cursor = conn.cursor()
    return conn, cursor


def mark_entry_as_treated(entryId, version=0):
    newVersion = version + 1
    conn, cursor = connect2Mysql()
    cursor.execute('update t_item set version = %s where id = %s', [
                   newVersion, entryId])
    insert_id = cursor.lastrowid
    conn.commit()
    cursor.close()


def fetch_entry_untreated(version=0):
    conn, cursor = connect2Mysql()
    cursor.execute(
        'select * from t_item where version = %s limit 0,1', (version,))
    values = cursor.fetchall()
    cursor.close()
    conn.close()

    if values[0] == None:
        print('||||||||||||||没有读取到|||||||||||||')
        return False
    else:
        print('||||||||||||||读取到一条未运算的文章|||||||||||||')
        return values[0]


def update_vec(entry, vec): #需要5ms
    vec = vec.tolist()
    vec = [round(v, 5) for v in vec]
    vec = json.dumps(vec)
    conn, cursor = connect2Mysql()
    cursor.execute(
        'update t_vocabulary set vector = %s where id = %s', [vec, entry['id']])
    insert_id = cursor.lastrowid
    conn.commit()
    cursor.close()


def getWordEntrys(word):
    conn, cursor = connect2Mysql()
    cursor.execute('select * from t_vocabulary where word = %s', (word,))
    values = cursor.fetchall()
    cursor.close()
    conn.close()
    return values


def insertVocabulary(word, startVector):
    startVector = [round(v, 5) for v in startVector]
    startStr = json.dumps(startVector)
    conn, cursor = connect2Mysql()
    cursor.execute('insert into t_vocabulary (word, vector) values (%s, %s)', [
                   word, startStr])
    insert_id = cursor.lastrowid
    conn.commit()
    cursor.close()
    return insert_id


def getNegSameples(contextWords, k=10):
    conn, cursor = connect2Mysql()
    cursor.execute('SELECT * from t_vocabulary ORDER BY RAND() LIMIT 0,100')
    values = cursor.fetchall()
    cursor.close()
    conn.close()
    return values
    
    # uniqueSamples = []
    # for entry in values:
    #     if entry[1] not in contextWords:
    #         uniqueSamples.append(entry)
    #         if len(uniqueSamples) == k:
    #             return uniqueSamples
    # return uniqueSamples



def getAll():
    conn, cursor = connect2Mysql()
    cursor.execute('select * from t_vocabulary')
    values = cursor.fetchall()
    cursor.close()
    conn.close()
    return values
def getWordById(entryId):
    conn, cursor = connect2Mysql()
    cursor.execute('select * from t_vocabulary where id = %s', (entryId,))
    values = cursor.fetchall()
    cursor.close()
    conn.close()
    return values[0]

def test(word):
    entrys = getWordEntrys(word)
    if len(entrys) == 0: print('没找到')
    cen_entry = entrys[0]
    allEntrys = getAll()
    unsortedList = []
    for et in allEntrys:
        deviationArr = np.array(json.loads(cen_entry[2])) - np.array(json.loads(et[2]))
        # deviationArr = np.fabs(deviationArr)
        deviationArr = [ round(de,3) for de in deviationArr.tolist()]
        deviationArr = np.square(np.array(deviationArr))
        deviation = np.sum(deviationArr)
        unsortedList.append({'deviation':deviation,'id':et[0]})
    sortedList = sorted(unsortedList, key=lambda dic: dic['deviation'])
    for nearId in sortedList[0:10]:
        print(getWordById(nearId['id'])[1])







# testGetNearest('人生')













