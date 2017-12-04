# -*- coding: utf-8 -*-

def _init():#初始化
    global _global_dict
    _global_dict = {}


def set(key,value):
    _global_dict[key] = value


def get(key,defValue=None):
    try:
        return _global_dict[key]
    except KeyError:
        return defValue
