import mysql.connector
import json
def connect2Mysql():
    conn = mysql.connector.connect(user='root', password='as56210', database='flow', use_unicode=True)
    cursor = conn.cursor()
    return conn, cursor
def fetch_entry_untreated(version='0'):
    conn, cursor = connect2Mysql()
    cursor.execute('select * from t_item where version = %s limit 0,1', (version,))
    values = cursor.fetchall()  
    cursor.close()
    conn.close()
    if values[0] == None:
        print('||||||||||||||没有读取到|||||||||||||')
        return False
    else:
        print('||||||||||||||读取到一条未运算的文章|||||||||||||')
        return values[0]

def mark_entry_as_treated(entryId,version=0):
    newVersion = version + 1
    conn, cursor = connect2Mysql()
    cursor.execute('update t_item set version = %s where id = %s', [newVersion, entryId])
    insert_id = cursor.lastrowid
    conn.commit()
    cursor.close()

def getWordEntry(word):
    conn, cursor = connect2Mysql()
    cursor.execute('select * from t_vocabulary where word = %s', (word,))
    values = cursor.fetchall()  
    cursor.close()
    conn.close()
    return values

def insertVocabulary(word,startVector):
    startStr = json.dumps(startVector)
    conn, cursor = connect2Mysql()
    cursor.execute('insert into t_vocabulary (word, vector) values (%s, %s)', [word, startStr])
    insert_id = cursor.lastrowid
    conn.commit()
    cursor.close()
    return insert_id

def getNegSameples(contextWords,k=10):
    conn, cursor = connect2Mysql()
    cursor.execute('SELECT * from t_vocabulary ORDER BY RAND() LIMIT 0,100')
    values = cursor.fetchall()  
    cursor.close()
    conn.close()

    uniqueSamples = []
    for entry in values:
        if entry[1] not in contextWords:
            uniqueSamples.append(entry)
            if len(uniqueSamples) == k: 
                print(uniqueSamples)
                return uniqueSamples
    print(uniqueSamples)
    return uniqueSamples

# getNegSameples(['订','一个','会'],3)
