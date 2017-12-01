# 返回未筛选过的 词组 数据
# 要筛选一些不用的词汇
# 需要一层逻辑 根据窗口大小 自动分配中心词与contextword

# 所以把分词单独搞一个模块，数据库操作单独放一个地方
# 1分词层，2筛选层，3根据窗口长度的训练对分配层

def do(arr,spanLength):
    returnArr = []
    c = spanLength    
    for index in range(len(arr)): 
        end = index + c if (index + 1 + c) <= len(arr) else len(arr)
        start = index - c if  (index + 1 - c) >= 0 else 0
        content = arr[start:end]
        item = {'center':arr[index],'context':content}
        returnArr.append(item)
    return returnArr
