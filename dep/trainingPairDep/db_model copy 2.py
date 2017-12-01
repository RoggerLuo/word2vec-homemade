#!/usr/bin/env python
# 导入MySQL驱动:
import mysql.connector
import jieba
user='root'
password='as56210'
database='flow'
def getUntreatedEntry_byVersion(version='0'):
    conn = mysql.connector.connect(user='root', password='as56210', database='flow', use_unicode=True)
    cursor = conn.cursor()
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

# 返回未筛选过的 词组 数据
# 要筛选一些不用的词汇
# 需要一层逻辑 根据窗口大小 自动分配中心词与contextword

# 所以把分词单独搞一个模块，数据库操作单独放一个地方
# 1分词层，2筛选层，3根据窗口长度的训练对分配层

def wordSegmentation():
    content = getUntreatedEntry_byVersion()
    seg_list = jieba.lcut(content[2])  # 默认是精确模式
    print('||||||||||||||分词|||||||||||||')
    print(seg_list)
    return seg_list
def getVectorEntry_byWord(word):
    conn = mysql.connector.connect(user='root', password='as56210', database='flow', use_unicode=True)
    cursor = conn.cursor()
    cursor.execute('select * from t_vocabulary where word = %s', (word,))
    values = cursor.fetchall()  
    cursor.close()
    conn.close()

    if len(values) == 0 : 
        return False
    else:
        return values[0]

print(getVectorEntry_byWord('abc'))
