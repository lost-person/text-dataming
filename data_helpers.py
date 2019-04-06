# coding = utf-8

'''
数据处理程序，包含：整合文件，数据预处理，创建数据集（交叉验证），创建batch
'''

import csv
import os
import random
import re
from collections import Counter

import jieba
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from utils import save_file, load_file, load_file2, load_file3
import word_vec

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

def seg2str(seg_data):
    '''
    将切分后的数据转换为字符串
    
    Args:
        seg_data object 切分后的数据
    Returns
        data str 字符串
    '''
    cont = ''
    line = ''
    for word in seg_data:
        if word == ' ':
            continue
        if word == '\n':
            line += word
            cont += line
            line = ''
            continue
        line += word + ' '
    return cont

def data_clean(neg_data_path, pos_data_path, des_path):
    '''
    数据预处理，包括分词，因为是短文本，就不去除数字，标点符号，停用词，低频词这些了
    
    Args:
        neg_data_path str 负面数据路径
        pos_data_path str 正面数据路径
        des_path str 存储路径
    '''
    # 读取合并文件
    neg_seg = jieba.cut(load_file(neg_data_path))
    pos_seg = jieba.cut(load_file(pos_data_path))
    # 将分词后的数据转换为字符串
    neg_data = seg2str(neg_seg)
    pos_data = seg2str(pos_seg)
    # 存储
    save_file(os.path.join(des_path, 'neg_clean.txt'), neg_data)
    save_file(os.path.join(des_path, 'pos_clean.txt'), pos_data)

def load_data_and_labels(neg_data_path, pos_data_path):
    '''
    读取数据集，并打上标签

    Args:
        neg_data_path str 负面数据路径
        pos_data_path str 正面数据路径
    Returns
        list 样本及其标签
    '''
    # 读取文件内容
    neg_data = load_file3(neg_data_path)
    pos_data = load_file3(pos_data_path)
    # 标签
    neg_labels = [[0, 1] for _ in neg_data]
    pos_labels = [[1, 0] for _ in pos_data]
    x = neg_data + pos_data
    y = np.concatenate([neg_labels, pos_labels], 0)
    return [x, y]

def make_dataset(x, y, n_splits = 10):
    '''
    使用交叉验证生成数据集

    Args:
        x array 数据
        y array 数据标签
        n_splits int 默认为10折
    Returns:
        train_index list 训练集索引
        test_index list 测试集索引
    '''
    kf = KFold(n_splits = n_splits)
    for train_index, test_index in kf.split(X = x, y = y):
        yield zip(x[train_index], x[test_index], y[train_index], y[test_index])

def batch_iter(data, batch_size, num_epochs, shuffle = True):
    """
    根据数据集生成batch

    Args:
        data array 数据
        batch_size int batch大小
        num_epochs int 训练轮数
        shuffle boolean 是否打乱数据
    """
    data = np.array(data)
    data_size = len(data)
    # 计算batch数
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for _ in range(num_epochs):
        # 每一轮打乱数据
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def load_pretrained_wv(wv_path, vocab_processor, embedding_dim):
    ''''
    载入预训练词向量，有的读取，没有的随机初始化

    Args:
        wv_path str 词向量路径
        vocab_processor obkject tensorflow的文本处理函数
        embedding_dim int 词向量维度
    Returns:
        embedding array 词向量
    '''
    model = word_vec.get_word_vec(wv_path)
    word_set = model.wv.index2word
    embedding = np.random.uniform(-1, 1, 
        size = (len(vocab_processor.vocabulary_), embedding_dim))
    word2id = vocab_processor.vocabulary_._mapping
    for word, index in word2id.items():
        if word in word_set:
            embedding[index, : ] = model[word]
    return embedding

def create_bert_dataset(neg_data_path, pos_data_path):
    '''
    创建bert的train_set, dev_set, test_set
    注：生成的是原始数据集，不含预处理操作
    '''
    # 载入数据，并添加标签
    neg_data = load_file3(neg_data_path)
    neg_data = [line.strip('\n') for line in neg_data]
    neg_label = [0 for _ in neg_data]
    pos_data = load_file3(pos_data_path)
    pos_data = [line.strip('\n') for line in pos_data]
    pos_label = [1 for _ in pos_data]
    x = neg_data + pos_data
    y = neg_label + pos_label

    train_data = []
    train_label = []
    dev_data = []
    dev_label = []
    test_data = []
    test_label = []

    # 生成数据集
    for i, line in enumerate(x):
        if random.random() < 0.1:
            dev_data.append(line)
            dev_label.append(y[i])
        elif random.random() < 0.2:
            test_data.append(line)
            test_label.append(0)
        else:
            train_data.append(line)
            train_label.append(y[i])

    # 存储
    data = {'label' : train_label, 'data' : train_data}
    train_df = pd.DataFrame(data)
    train_df.to_csv(path_or_buf = os.path.join(neg_data_path, '..', 'train.tsv'), sep = '\t', index = False)
    data = {'label' : dev_label, 'data' : dev_data}
    dev_df = pd.DataFrame(data)
    dev_df.to_csv(path_or_buf = os.path.join(neg_data_path, '..', 'dev.tsv'), sep = '\t', index = False)
    data = {'label' : test_label, 'data' : test_data}
    test_df = pd.DataFrame(data)
    test_df.to_csv(path_or_buf = os.path.join(neg_data_path, '..', 'test.tsv'), sep = '\t', index = False)

if __name__ == '__main__':
    # merge_files('./data/, './data')
    # data_clean('./data/Book_del_4000/neg.txt', './data/Book_del_4000/pos.txt', './data/Book_del_4000/')
    # load_data_and_labels('./data/htl_del_4000/neg_clean.txt', './data/htl_del_4000/pos_clean.txt')
    create_bert_dataset('./data/NB_del_4000/neg.txt', './data/NB_del_4000/pos.txt')