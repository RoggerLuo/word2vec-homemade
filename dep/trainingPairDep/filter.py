#!/usr/bin/env python

ignoreds = [
    '，',',','的','是'
]
def do(arr):
    print('||||||||||||||筛选过滤|||||||||||||')
    filteredArr = []
    for igs in arr:
        if igs not in ignoreds:
            filteredArr.append(igs)
    return filteredArr


