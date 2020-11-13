from DataGenerator import DataGenerator
import pandas as pd
import itertools
import numpy as np
import random
import csv
import time

import matplotlib.pyplot as plt

import tensorflow as tf

import tensorflow.keras.backend as K
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout


class EmbeddingsGenerator:
    def __init__(self, train_users, data):
        print('train_users: ', train_users)
        print('data: ', data)
        self.train_users = train_users
        self.data = data
        # preprocess
        self.data = data.sort_values(by=['timestamp'])
        #self.data = sorted(data, key='timestamp')
        # make them start at 0
        # 유저아이디 인트로 인덱싱해줘야함

        user_uniq = self.data['user'].unique()
        print('user_uniq: ', user_uniq)
        user_ix = [num for num in range(len(user_uniq))]
        #user_ix_uniq = dict(zip(user_ix, user_uniq))
        user_uniq_ix = dict(zip(user_uniq, user_ix))
        self.data.replace({"user": user_uniq_ix})
    
        item_uniq = self.data['item'].unique()
        item_ix = [num for num in range(len(item_uniq))]
        #user_ix_uniq = dict(zip(user_ix, user_uniq))
        item_uniq_ix = dict(zip(item_uniq, item_ix))
        self.data.replace({"item": item_uniq_ix})
    
        self.data['user'] = self.data['user'] 
        self.data['item'] = self.data['item'] 
        #self.user_count = self.data['user'].max() + 1
        #self.book_count = self.data['item'].max() + 1
        self.user_count = len(self.data['user'].unique())
        self.book_count = len(self.data['item'].unique())
        self.user_list = self.data['user'].unique()
        # list of rated books by each user
        self.user_books = {}
        print('user: ', self.data['user'].unique())
        print('book_count: ', self.book_count)
        print('user_count: ', self.user_count)
        
        for user in self.user_list:
            print('user idx: ', user)
            self.user_books[user] = self.data[self.data.user == user]['item'].tolist()
        self.m = self.model()
    
    def model(self, hidden_layer_size=100):
        m = Sequential()
        m.add(Dense(hidden_layer_size, input_shape=(1, self.book_count)))
        m.add(Dropout(0.2))
        m.add(Dense(self.book_count, activation='softmax'))
        m.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return m
    
    def generate_input(self, user):
        '''
        Returns a context and a target for the user
        
        context: user's history with one random book removed
        target: id of random removed book
        '''
        user_books_count = len(self.user_books[user])
        # picking random book
        random_index = np.random.randint(0, user_books_count - 1) # -1 avoids taking the las book
        # setting target
        target = np.zeros((1, self.book_count))
        target[0][self.user_books[user][random_index]] = 1
        # setting context
        context = np.zeros((1, self.book_count))
        context[0][self.user_books[user][:random_index] + self.user_books[user][random_index + 1:]] = 1
        return context, target
    
    def train(self, nb_epochs=300, batch_size=10000):
        '''
        Trains the model from train_users's history
        '''
        for i in range(nb_epochs):
            print('%d/%d' % (i+1, nb_epochs))
            #batch = [self.generate_input(user=np.random.choice(self.train_users) - 1) for _ in range(batch_size)]
            batch = [self.generate_input(user=np.random.choice(self.train_users)) for _ in range(batch_size)]
            X_train = np.array([b[0] for b in batch])
            y_train = np.array([b[0] for b in batch])
            self.m.fit(X_train, y_train, epochs=1, validation_split=0.5)
        
    def test(self, test_users, batch_size=10000):
        '''
        Returns [loss, accuracy] on the test set
        '''
        batch_test = [self.generate_input(user=np.random.choice(test_users) - 1) for _ in range(batch_size)]
        X_test = np.array([b[0] for b in batch_test])
        y_test = np.array([b[1] for b in batch_test])
        return self.m.evaluate(X_test, y_test)
    
    def save_embeddings(self, file_name):
        '''
        Generates a csv file containing the vecotr embedding for each book
        '''
        inp = self.m.input                                          # input placeholder
        outputs = [layer.output for layer in self.m.layers]         # all layer outputs
        functor = K.function([inp, K.learning_phase()])             # evaluation function
        
        # append embeddings to vectors
        vectors = []
        for book_id in range(self.book_count):
            book = np.zeros((1, 1, self.book_count))
            book[0][0][book_id] = 1
            layer_outs = fuctor([book])
            vector = [str(v) for v in layer_outs[0][0][0]]
            vector = '|'.join(vector)
            vectors.append([book_id, vector])
        
        # saves as a csv file
        embeddings = pd.DataFrame(vectors, columns=['item_id', 'vectors']).astype({'book_id': 'int32'})
        embeddings.to_csv(file_name, sep=';', index=False)
        files.download(file_name)

data_path ='./book_up3k.csv'
data = pd.read_csv(data_path, names={'item': int, 'user': str, 'rating': int, 'timestamp': int}, header=0)

dg = DataGenerator(data_path)
dg.gen_train_test(0.8, seed=42)

eg = EmbeddingsGenerator(dg.user_train, data)

eg.train(nb_epochs=10, batch_size=16)
train_loss, train_accuracy = eg.test(df.user_train)
print('Train set: Loss=%.4f ; Accuracy=%.1f%%' % (train_loss, train_accuracy * 100))
test_loss, test_accuracy = eg.test(dg.user_test)
print('Test set; Loss=%.4f; Accuracy=%.1f%%' % (test_loss, test_accuracy * 100))
eg.save_embeddings('embeddings.csv')