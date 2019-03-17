# coding = utf-8

'''
数据处理程序，包含：整合文件，数据预处理，创建数据集（交叉验证），创建batch
'''

import os
import random
import re
import time
from collections import Counter

import jieba
from sklearn.model_selection import KFold
from utils import save_file, load_file, load_file2

def merge_files(src_path, des_path):
    '''
    读取整合文件——将多个文件去除空行，并整合到一个文件当中，一个文件占据一行。
    注意：源数据文件编码方式为GBK，整合后的文件为utf-8

    Args:
        src_path str 源数据路径
        des_path str 目的数据路径，用于存储整合后的文件
    '''
    if not os.path.exists(src_path):
        raise FileNotFoundError

    # 遍历指定目录
    for root, _, files in os.walk(src_path):
        if len(files) == 0:
            continue
        # 文本内容
        cont = ''
        for data_file in files:
            cont += load_file2(os.path.join(root, data_file), encoding = 'gb18030') + '\n'
        # 存储整合文件
        merge_file_path = root + '.txt'
        save_file(merge_file_path, cont)

def data_clean():
    pass

def make_dataset():
    pass

def batch_iter():
    pass

if __name__ == '__main__':
    merge_files('./data', './data')
