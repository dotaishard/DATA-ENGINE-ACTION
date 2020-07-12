# -*- coding:utf-8 -*-
# 词云展示
from wordcloud import WordCloud
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from lxml import etree
from nltk.tokenize import word_tokenize

# 数据加载
data = pd.read_csv('./Market_Basket_Optimisation.csv', header = None)
# 将数据中nan替换为'none'
data = data.fillna('none')
# 整合字段
temp = data.values[0,0]
for i in range(0, data.shape[0]):
    for j in range(0, 20):
        temp = temp + "," + data.values[i,j]

# 去掉停用词'none'
def remove_stop_words(f):
	stop_words = ['none']
	for stop_word in stop_words:
		f = f.replace(stop_word, '')
	return f

# 生成词云
def create_word_cloud(f):
	print('根据词频，开始生成词云!')
	f = remove_stop_words(f)
	cut_text = word_tokenize(f)
	#print(cut_text)
	cut_text = ",".join(cut_text)
	wc = WordCloud(
		max_words=10,
		width=2000,
		height=1200,
    )
	wordcloud = wc.generate(cut_text)
	# 写词云图片
	wordcloud.to_file("wordcloud.jpg")
	# 显示词云文件
	plt.imshow(wordcloud)
	plt.axis("off")
	plt.show()



# 生成词云
create_word_cloud(temp)