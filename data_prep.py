import pandas as pd
import numpy as np
import random
import csv
import time

import matplotlib.pyplot as plt

import tensorflow as tf

import keras.backend as K
from keras import Sequential
from keras.layers import Dense, Dropout

data = pd.read_csv('./data/Books.csv', names=['item', 'user', 'rating', 'timestamp'])

# User: str to int
data.user = pd.Categorical(data.user)
data['user'] = data.user.cat.codes

# Item: str to int
data.item = pd.Categorical(data.item)
data['item'] = data.item.cat.codes

def history_upper20(data):
    a = pd.DataFrame(data['user'].value_counts())
    user_indices = a[a['user']>=20].index 
    data_20 = data[data['user'].isin(user_indices)]
    
    return data_20



def item_upper10(data):
    item_counts = pd.DataFrame(data['item'].value_counts())
    item_indices = item_counts[item_counts['item']>=10].index 
    data_10 = data[data['item'].isin(item_indices)]
    return data_10

def rating_upper3(data):
    rating_indices = data['rating']>3
    data_3 = data[rating_indices]
    return data_3


def preprocess(data, name):
    data_10 = item_upper10(data)
    data_20_10 = history_upper20(data_10)
    latest_data = data_20_10.drop_duplicates(['user','item'],keep='first')

    usercounts = pd.DataFrame(latest_data['user'].value_counts())
    itemcounts = pd.DataFrame(latest_data['item'].value_counts())

    print('shape: ', latest_data.shape)
    print('user counts: ', usercounts)
    print('item counts: ', itemcounts)

    latest_data.to_csv('./data/' + name, index=False, header=None)
    return latest_data
