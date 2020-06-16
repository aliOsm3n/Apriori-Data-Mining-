# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 23:13:53 2017

@author: AliOthman
"""

import pandas as pd
from mlxtend.preprocessing import OnehotTransactions
from mlxtend.frequent_patterns import apriori


#dataset = [['Milk', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
#           ['Dill', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
#           ['Milk', 'Apple', 'Kidney Beans', 'Eggs'],
#           ['Milk', 'Unicorn', 'Corn', 'Kidney Beans', 'Yogurt'],
#           ['Corn', 'Onion', 'Onion', 'Kidney Beans', 'Ice cream', 'Eggs']]

data = pd.read_csv('bakery.csv')
process = OnehotTransactions()
matrix = process.fit(data).transform(data)

df = pd.DataFrame(matrix, columns=process.columns_)

#apriori(df, min_support=0.6)
#apriori(df, min_support=0.6, use_colnames=True)

frequent_itemsets = apriori(df, min_support=0.0123, use_colnames=True)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))

#frequent_itemsets[ (frequent_itemsets['length'] == 2) &
#                  (frequent_itemsets['support'] >= 0.8) ]

print frequent_itemsets