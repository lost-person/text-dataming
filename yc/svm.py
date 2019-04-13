#coding=utf-8
import jieba
import os
import chardet
import re
from langconv import *
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import naive_bayes
import numpy as np
import random

def normalize(str):
    str=str.strip()
    str=re.sub(r'[0-9]\)', "", str, count=0, flags=0)
    str = re.sub(r'[0-9]．', "", str, count=0, flags=0)
    str = re.sub(r'[0-9] ', "", str, count=0, flags=0)
    str = re.sub(r'[0-9]{0,4}年', "", str, count=0, flags=0)
    str = re.sub(r'[0-9]{0,2}月', "", str, count=0, flags=0)
    str = re.sub(r'[0-9]{0,2}日', "", str, count=0, flags=0)
    str = str.replace("免费注册 网站导航 宾馆索引 服务说明 关于携程 诚聘英才 代理合作 广告业务 联系我们","")
    str = str.replace("免费注册 网站导航 宾馆索引 服务说明 关于", "")
    str = str.replace("免费注册 网站导航 宾馆索引 服务说明 关于携程 诚聘英", "")
    str = str.replace("ntent>", "")
    str = re.sub(r'[0-9]{0,4}com', "", str, count=0, flags=0)
    str = str.replace("all rights reserved","")
    str = str.strip("Copyright 19992008 ctripcom all rights reserved")
    str = re.sub(r'[0-9]{0,4} ctrip[a-z]{0,3}', "", str, count=0, flags=0)

    str = re.sub(r'下一页', "", str, count=0, flags=0)
    for i in ["，", "；", "。", ".", "！", "`", "：", "~", "（", "）", "-", ",", "、", "!", "\"", "?", "？", "*", "“", "”","…",":","(",")","．","＂","～","－",";","【","】","﹗",">>",">","<","《","》"]:
        str = str.replace(i, "")
    return str

ltneg=os.listdir("./neg")
ltpos=os.listdir("./pos")
trainset = []
testset = []
dic_unigram = {}
dic_bigram = {}
look_up_unigram={}
look_up_bigram={}
trainset = []
testset = []

# 构建消极词典
for index in range(2000):
    f = open("./neg/"+ltneg[index],"r",encoding='GB2312')
    st = ""
    try:
        lines = f.readlines()
        for line in lines:
            line = Converter('zh-hans').convert(line)
            line = normalize(line)
            st = st + line
    except:
        f = open("./neg/" + ltneg[index], "r", encoding='GBK',errors='ignore')
        lines = f.readlines()
        for line in lines:
            line = Converter('zh-hans').convert(line)
            line = normalize(line)
            st = st + line
    result = jieba.cut(st)
    finalresult = []
    for r in result:
        if " "==r:
            continue
        finalresult.append(r)
    if index >= 1800:
        testset.append([finalresult,1])
    else:
        trainset.append([finalresult,1])
    for i in range(len(finalresult)):
        if finalresult[i] in dic_unigram:
            dic_unigram[finalresult[i]] += 1
        else:
            dic_unigram[finalresult[i]] = 1
        if i+1<len(finalresult):
            if (finalresult[i]+finalresult[i+1]) in dic_bigram:
                dic_bigram[finalresult[i]+finalresult[i+1]] +=1
            else:
                dic_bigram[finalresult[i] + finalresult[i + 1]] = 1

# 构建积极词典
for index in range(2000):
    f = open("./pos/"+ltpos[index],"r",encoding='GB2312')
    st = ""
    if ltpos[index]=="pos.907.txt":
        continue
    try:
        lines = f.readlines()
        for line in lines:
            line = Converter('zh-hans').convert(line)
            line = normalize(line)
            st = st + line
    except:
        f = open("./pos/" + ltpos[index], "r", encoding='GBK',errors='ignore')
        lines = f.readlines()
        for line in lines:
            line = Converter('zh-hans').convert(line)
            line = normalize(line)
            st = st + line
    result = jieba.cut(st)
    finalresult = []
    for r in result:
        if " "==r:
            continue
        finalresult.append(r)
    if index >= 1800:
        testset.append([finalresult,-1])
    else:
        trainset.append([finalresult,-1])
    for i in range(len(finalresult)):
        if finalresult[i] in dic_unigram:
            dic_unigram[finalresult[i]] += 1
        else:
            dic_unigram[finalresult[i]] = 1
        if i+1<len(finalresult):
            if (finalresult[i]+finalresult[i+1]) in dic_bigram:
                dic_bigram[finalresult[i]+finalresult[i+1]] +=1
            else:
                dic_bigram[finalresult[i] + finalresult[i + 1]] = 1
# 构建词典
index = 0
for i in dic_unigram:
    if dic_unigram[i] >= 3:
        look_up_unigram[i] = index
        index += 1

index = 0
for i in dic_bigram:
    if dic_bigram[i] >= 4:
        look_up_bigram[i] = index
        index += 1

print("词典构造完毕")
print("unigram词典大小:"+str(len(look_up_unigram)))
print("bigram词典大小:"+str(len(look_up_bigram)))

# 构造训练集和测试集特征(unigram)
traindata_unigram = []
trainlabel = []
testdata_unigram = []
testlabel = []

# 打乱顺序
random.shuffle(trainset)
random.shuffle(testset)

for tr in trainset:
    trainlabel.append(tr[1])
    temp = [0 for i in range(len(look_up_unigram))]
    #temp = [0 for i in range(len(look_up_bigram))]
    #temp = [0 for i in range(len(look_up_unigram)+len(look_up_bigram))]
    for t in range(len(tr[0])):
        if tr[0][t] in look_up_unigram:
           temp[look_up_unigram[tr[0][t]]] = 1
        # if t+1 < len(tr[0]) and (tr[0][t]+tr[0][t+1]) in look_up_bigram:
        #     temp[look_up_bigram[tr[0][t]+tr[0][t+1]]] = 1
        # if tr[0][t] in look_up_unigram:
        #    temp[look_up_unigram[tr[0][t]]] = 1
        # if t+1 < len(tr[0]) and (tr[0][t]+tr[0][t+1]) in look_up_bigram:
        #     temp[len(look_up_unigram)+look_up_bigram[tr[0][t]+tr[0][t+1]]] = 1
    traindata_unigram.append(temp)
print("训练集构造完毕")

for tr in testset:
    testlabel.append(tr[1])
    temp = [0 for i in range(len(look_up_unigram))]
    #temp = [0 for i in range(len(look_up_bigram))]
    #temp = [0 for i in range(len(look_up_unigram) + len(look_up_bigram))]
    for t in range(len(tr[0])):
        if tr[0][t] in look_up_unigram:
          temp[look_up_unigram[tr[0][t]]] = 1
        # if t+1 < len(tr[0]) and (tr[0][t]+tr[0][t+1]) in look_up_bigram:
        #     temp[look_up_bigram[tr[0][t]+tr[0][t+1]]] = 1
        # if tr[0][t] in look_up_unigram:
        #    temp[look_up_unigram[tr[0][t]]] = 1
        # if t+1 < len(tr[0]) and (tr[0][t]+tr[0][t+1]) in look_up_bigram:
        #     temp[len(look_up_unigram)+look_up_bigram[tr[0][t]+tr[0][t+1]]] = 1
    testdata_unigram.append(temp)
print("测试集构造完毕")

#clf=RandomForestClassifier(max_depth=10, random_state=0,n_estimators=5,criterion="entropy")
#clf=svm.SVC(C=0.1, kernel='linear')
clf=LogisticRegression(penalty="l2",tol=0.1,C=1)
#clf=naive_bayes.GaussianNB()

clf.fit(traindata_unigram,trainlabel)
print("训练完成")

tol=0
res=clf.predict(testdata_unigram)
for i in range(0,len(testlabel)):
	if res[i]==testlabel[i]:
		tol+=1
print("accuracy:"+str(tol*1.0/len(testlabel)))