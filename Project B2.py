# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 16:21:11 2020

@author: bibiboom
"""



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from efficient_apriori import apriori

data = pd.read_csv('dingdanbiao2.csv', encoding = 'gbk')
data = data[["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q"]].drop(columns = ['a','b','c','d','e','g','h','i','j','k','l','m','n','p','q']).sort_values(by="o",ascending=True) 
print (data)
transactions = []
for i in range(0, data.shape[0]):
    temp = []
    for j in range(0, 2):
        if str(data.values[i, j]) != 'nan':
           temp.append(str(data.values[i, j]))
    transactions.append(temp)
#print(transactions)
itemsets, rules = apriori(transactions, min_support=0.02,  min_confidence=0.2)
print("频繁项集：", itemsets)
print("关联规则：", rules)