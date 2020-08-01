# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 16:21:11 2020

@author: bibiboom
"""



import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

data = pd.read_csv('dingdanbiao2.csv', encoding = 'gbk')
data = data[["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q"]]
data = data.drop(columns = ['a','b','c','d','e','g','h','i','j','k','m','n','o','p','q'])
data_HE = data.drop('f',1).join(data.f.str.get_dummies(sep='|'))
#print (data_HE)
#data_HE.to_csv("data_HE.csv",index=False)
data_HE.set_index(['l'],inplace=True)
data_HE = data_HE.sort_values(by="l" , ascending=True) 
data_HE= data_HE.groupby(['l']).agg(['max'])
#print (data_HE)
itemsets = apriori(data_HE,use_colnames=True, min_support=0.05)
itemsets = itemsets.sort_values(by="support" , ascending=False) 
print (itemsets)
rules = association_rules(itemsets, metric='lift', min_threshold=1)
rules = rules.sort_values(by="lift" , ascending=False) 
rules.to_csv('Project B.csv')
print (rules)
