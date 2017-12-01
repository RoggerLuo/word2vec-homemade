import jieba

ignoreds = [
    '，',',','的','是'
]

def segment(string):
    seg_list = jieba.lcut(string)  # 默认是精确模式
    print('||||||||||||||分词|||||||||||||')
    print(seg_list)
    return seg_list

def filter(arr):
    print('||||||||||||||筛选过滤|||||||||||||')
    filteredArr = []
    for igs in arr:
        if igs not in ignoreds:
            filteredArr.append(igs)
    return filteredArr

def slide(arr,spanLength):
    print('||||||||||||||滑窗,步长为|||||||||||||')
    print(spanLength)
    returnArr = []
    c = spanLength    
    for index in range(len(arr)): 
        end = index + c if (index + 1 + c) <= len(arr) else len(arr)
        start = index - c if  (index + 1 - c) >= 0 else 0
        content = arr[start:end]
        item = {'center':arr[index],'context':content}
        returnArr.append(item)
    return returnArr
