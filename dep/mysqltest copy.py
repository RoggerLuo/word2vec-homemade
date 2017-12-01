#!/usr/bin/env python

# 导入MySQL驱动:
import mysql.connector
user='root'
password='as56210'
database='flow'

# 注意把password设为你的root口令:
def test():
    conn = mysql.connector.connect(user='root', password='as56210', database='flow', use_unicode=True)

    # cursor = conn.cursor()
    # 创建user表:
    # cursor.execute('create table user (id varchar(20) primary key, name varchar(20))')
    # 插入一行记录，注意MySQL的占位符是%s:
    # cursor.execute('insert into user (id, name) values (%s, %s)', ['1', 'Michael'])
    # cursor.rowcount
    # 提交事务:
    # conn.commit()
    # cursor.close()

    # 运行查询:
    cursor = conn.cursor()
    cursor.execute('select * from flow_item where id = %s', ('563',))
    values = cursor.fetchall()
    print(values)
    # 关闭Cursor和Connection:
    cursor.close()
    conn.close()
