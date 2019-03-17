# coding = utf-8

import os

def load_file(path, encoding = 'utf-8'):
    '''
    读取文件内容

    Args:
        path str 数据读取路径
        encoding str 编码方式
    Returns:
        读取内容
    '''
    if not os.path.exists(path):
        raise FileNotFoundError
    with open(path, 'r', encoding = encoding) as f:
        return f.read()

def load_file2(path, encoding = 'utf-8'):
    '''
    读取文件内容，删除空行及Book_del_4000中的content>

    Args:
        path str 数据读取路径
        encoding str 编码方式
    Returns:
        读取内容
    '''
    if not os.path.exists(path):
        raise FileNotFoundError
    cont = ''
    with open(path, 'r', encoding = encoding) as f:
        for line in f.readlines():
            if line.find('content>') != -1 or line.find('下一页>>') != -1 or line.find('免费注册') != -1 or line.find('Copyright@') != -1:
                continue
            cont += line.strip()
    return cont

def load_file3(path, encoding = 'utf-8'):
    '''
    以行的形式读取文件

    Args:
        path str 数据读取路径
        encoding str 编码方式
    Returns:
        读取内容
    '''
    if not os.path.exists(path):
        raise FileNotFoundError
    with open(path, 'r', encoding = encoding) as f:
        return f.readlines()

def save_file(path, cont, encoding = 'utf-8'):
    '''
    存储文本数据

    Args:
        path str 存储路径
        cont object 存储内容
        encoding str 文本编码方式 
    '''
    with open(path, 'w', encoding = encoding) as f:
        f.write(cont)

if __name__ == "__main__":
    pass