import db_model
import jieba
import json
import numpy as np

dimVectors = 8
ignoreds = ['，', ',', '的', '是', '\n', ' ']


def getStartVector():
    randomStartVector = (np.random.rand(dimVectors) - 0.5)
    zerosVector = np.zeros((dimVectors))
    wordVectors = np.concatenate(
        (randomStartVector / dimVectors, zerosVector), axis=0)
    # print(wordVectors)
    # wordVectors = wordVectors.astype('float16')
    wordVectors = wordVectors.tolist()
    wordVectors = [round(vector, 5) for vector in wordVectors]
    # print(wordVectors)
    return wordVectors


def segment(string):
    seg_list = jieba.lcut(string)  # 默认是精确模式
    # print('||||||||||||||分词|||||||||||||')
    return seg_list


def filterWord(arr):
    # print('||||||||||||||筛选过滤|||||||||||||')
    filteredArr = []
    for word in arr:
        if word not in ignoreds:
            filteredArr.append(word)
    # print(filteredArr)
    return filteredArr


def getIdAndVector(word):
    entrys = db_model.getWordEntrys(word)
    if len(entrys) == 0:
        startVector = getStartVector()
        insert_id = db_model.insertVocabulary(word, startVector)
        return insert_id, startVector
    else:
        vectorFetched = entrys[0][2]
        vectorFetched = json.loads(vectorFetched)
        entry_id = entrys[0][0]
        return entry_id, vectorFetched


def getDataset(string, windowLength=10):
    wordList = segment(string)
    arr = filterWord(wordList)
    # print('||||||||||||||滑窗,步长为|||||||||||||')
    # print(windowLength)
    tokens = {}
    trainingPairs = []
    wordVectors = {}

    c = windowLength

    for index in range(len(arr)):
        word = arr[index]
        start = index - c if (index - c) >= 0 else 0
        end = index + 1 + c if (index + 1 + c) <= len(arr) else len(arr)

        content = arr[start:index]
        content2 = arr[index + 1:end]
        content.extend(content2)
        # item = {'center':arr[index],'context':content}
        item = (word, content)
        trainingPairs.append(item)
        # -------
        insert_id, vec = getIdAndVector(word)
        tokens[word] = insert_id
        wordVectors[word] = vec
    return trainingPairs, tokens, wordVectors


# entry = db_model.fetch_entry_untreated()
# ps, tks, vec = getDataset(entry[2], 3)
# db_model.mark_entry_as_treated(entry[0])
# print(ps)
# print(tks)
# print(vec)


# # content = getUntreatedEntry_byVersion()
# # wordList = segment(content[2])
# # wordList = filterWord(wordList)
# # test = slide(wordList,5)
# # print(test[2])
