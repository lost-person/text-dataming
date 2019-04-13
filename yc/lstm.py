#coding=utf-8
import jieba
import os
import chardet
import re
from langconv import *
import numpy as np
import random
import torch
from torch import nn,optim
import torch.nn.functional as F
from torch.autograd import Variable
from zhon.hanzi import punctuation
import string
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence


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
	for i in punctuation:
		str = str.replace(i, "")
	for i in string.punctuation:
		str = str.replace(i, "")
	return str

class LSTM(nn.Module):
	def __init__(self,word_size,hidden_size,n_layer,voc_size,batch_size):
		super(LSTM,self).__init__()
		self.n_layer=n_layer
		self.word_size=word_size
		self.hidden_size=hidden_size
		self.voc_size=voc_size
		self.batch_size=batch_size
		self.dropout=nn.Dropout(0.3)
		self.lstm=nn.LSTM(input_size=word_size,
			hidden_size=hidden_size,
			num_layers=n_layer,
			bidirectional=True,
			batch_first=True,
		)
		weight = torch.randn(self.voc_size,self.word_size)
		self.layer=nn.Linear(256,2)
		for v in look_up_unigram:
			try:
				index=look_up_unigram[v]
				weight[index, :] = torch.from_numpy(w2v[v])
			except:
				continue
		self.word_embeddings = nn.Embedding.from_pretrained(weight)
		self.word_embeddings.weight.requires_grad = True

	# 隐藏层初始化
	def init_hidden(self,bs):
		return (Variable(torch.zeros(2, bs, self.hidden_size)),Variable(torch.zeros(2, bs, self.hidden_size)))

	def forward(self,r_input,bs):
		hidden=self.init_hidden(bs)
		embeds=self.word_embeddings(r_input)
		r_out,self.hidden=self.lstm(embeds.view(bs,-1,self.word_size),hidden)
		out=r_out[:,-1,:]
		out=self.dropout(out)
		out=self.layer(out)
		return out



ltneg=os.listdir("./neg")
ltpos=os.listdir("./pos")
trainset = []
testset = []
dic_unigram = {}
look_up_unigram={}
trainset = []
testset = []
lookup_label={}
lookup_label["neg"]=0
lookup_label["pos"]=1

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
		testset.append([finalresult,lookup_label["neg"]])
	else:
		trainset.append([finalresult,lookup_label["neg"]])
	for i in range(len(finalresult)):
		if finalresult[i] in dic_unigram:
			dic_unigram[finalresult[i]] += 1
		else:
			dic_unigram[finalresult[i]] = 1

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
		testset.append([finalresult,lookup_label["pos"]])
	else:
		trainset.append([finalresult,lookup_label["pos"]])
	for i in range(len(finalresult)):
		if finalresult[i] in dic_unigram:
			dic_unigram[finalresult[i]] += 1
		else:
			dic_unigram[finalresult[i]] = 1
		

# 构建词典
index = 0
for i in dic_unigram:
	if dic_unigram[i] >= 0:
		look_up_unigram[i] = index
		index += 1

look_up_unigram["<unk>"] = index
index += 1
look_up_unigram["<padding>"] = index
index += 1
voc_size=len(look_up_unigram)
print("词典构造完毕")
print("unigram词典大小:"+str(len(look_up_unigram)))



# 构造训练集和测试集特征(unigram)
traindata_unigram = []
trainlabel = []
testdata_unigram = []
testlabel = []

# 打乱顺序
random.shuffle(testset)
for tr in testset:
	testlabel.append(tr[1])
	temp = []
	for t in range(len(tr[0])):
		if tr[0][t] in look_up_unigram:
			temp.append(look_up_unigram[tr[0][t]])
		else:
			temp.append(look_up_unigram["<unk>"])
	testdata_unigram.append(temp)
print("测试集构造完毕")

#加载词向量
w2v=Word2Vec.load("./w2v.model")

# 定义损失函数和优化器
# 词向量维度128，隐藏层大小128，batch size 64，epoch 5
criterion=nn.CrossEntropyLoss()
model=LSTM(128,128,1,voc_size,64)
optimizer=optim.Adam(model.parameters(),lr=1e-3)
print("参数初始化完毕")

maxlen=0
median_data=[]
median_label=[]
#定义训练
for epoch in range(25):
	random.shuffle(trainset)
	traindata_unigram=[]
	trainlabel=[]
	for tr in trainset:
		trainlabel.append(tr[1])
		temp = []
		for t in range(len(tr[0])):
			if tr[0][t] in look_up_unigram:
				temp.append(look_up_unigram[tr[0][t]])
			else:
				temp.append(look_up_unigram["<unk>"])
		traindata_unigram.append(temp)
	print("训练集重排完毕")

	for index in range(len(traindata_unigram)):
		if (index+1)%64==0 or index==len(traindata_unigram)-1:
			if len(traindata_unigram[index])>maxlen:
				maxlen=len(traindata_unigram[index])
			median_data.append(traindata_unigram[index])
			median_label.append(trainlabel[index])
			if index==len(traindata_unigram)-1:
				bs=len(traindata_unigram)%64
			else:
				bs=64
			for i in range(len(median_data)):
				if len(median_data[i])<maxlen:
					for _ in range(maxlen-len(median_data[i])):
						median_data[i].append(look_up_unigram["<padding>"])
			inpd=Variable(torch.LongTensor(median_data))
			inpl=Variable(torch.LongTensor(median_label))
			out=model(inpd,bs)
			loss=criterion(out,inpl)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			print("epoch:"+str(epoch)+" data:"+str(index+1)+" loss"+str(loss))		
			maxlen=0
			median_data=[]
			median_label=[]
		else:
			if len(traindata_unigram[index])>maxlen:
				maxlen=len(traindata_unigram[index])
			median_data.append(traindata_unigram[index])
			median_label.append(trainlabel[index])
print("训练完成")

# 模型保存
torch.save(model.state_dict(),'model.pkl')

#定义测试
correct = 0
index = 0
for	index in range(len(testdata_unigram)):
	if (index+1)%64==0 or index==len(testdata_unigram)-1:
		if len(testdata_unigram[index])>maxlen:
			maxlen=len(testdata_unigram[index])
		median_data.append(testdata_unigram[index])
		median_label.append(testlabel[index])
		if index==len(testdata_unigram)-1:
			bs=len(testdata_unigram)%64
		else:
			bs=64
		for i in range(len(median_data)):
			if len(median_data[i])<maxlen:
				for _ in range(maxlen-len(median_data[i])):
					median_data[i].append(look_up_unigram["<padding>"])
		inpd=Variable(torch.LongTensor(median_data))
		out=model(inpd,bs)
		predict=torch.argmax(out,1)
		for p in range(len(predict)):
			print(predict[p])
			if predict[p] == testlabel[p+index+1-bs]:
				correct+=1
		maxlen=0
		median_data=[]
		median_label=[]	
	else:
		if len(testdata_unigram[index])>maxlen:
			maxlen=len(testdata_unigram[index])
		median_data.append(testdata_unigram[index])
		median_label.append(testlabel[index])
print("accuracy: "+str(correct*1.0/len(testlabel)))