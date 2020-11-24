import itertools
import pandas as pd
import numpy as np
import random
import csv
import time

import matplotlib.pyplot as plt

import tensorflow as tf

import tensorflow.keras.backend as K
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout

class DataGenerator():
    def __init__(self, datapath):
        self.data = self.load_datas(datapath)
        self.users = self.data['user'].unique()
        self.items = self.data['item'].unique()
        self.histo = self.gen_histo()
        self.train = []
        self.test = []
        #self.data_small
        
    
    def load_datas(self, datapath):
        #data = pd.read_csv(datapath, names=['item', 'user', 'rating', 'timestamp'])
        data = pd.read_csv(datapath, dtype={'item': str, 'user': str, 'rating': int, 'timestamp': int}, header=0)
        #data2 = data.astype({'item': str, 'user': str, 'rating': int, 'timestamp': int})
        #data['item'] = data['item'].astype(str)
        #data['user'] = data['user'].astype(str)
        #data['rating'] = data['rating'].astype(str)
        #data['rating'] = data['rating'].astype(int)
        #data['timestamp'] = data['timestamp'].astype(str)
        #data['timestamp'] = data['timestamp'].astype(int)

        #print(type(data2['item']), type(data2['user']), type(data2['rating']), type(data2['timestamp']))

        #data2.head()

        return data
    
    def gen_histo(self):
        historic_users = []
        for i, u in enumerate(self.users):
            #self.data_small = self.data[self.data['user'] == u]
            temp = self.data[self.data['user'] == u]
            temp = temp.sort_values('timestamp').reset_index()
            temp.drop('index', axis=1, inplace=True)
            historic_users.append(temp)
        return historic_users
    
    def sample_histo(self, user_histo, action_ratio=0.8, 
                     max_samp_by_user=5, max_state=100, max_action=50, nb_states=[], nb_actions=[]):
        n = len(user_histo)
        
        sep = int(action_ratio * n)
        nb_sample = random.randint(1, max_samp_by_user)

        if not nb_states:
            nb_states = [min(random.randint(1, sep), max_state) for i in range(nb_sample)]
        if not nb_actions:
            nb_actions = [min(random.randint(1, n-sep), max_action) for i in range(nb_sample)]
        
        assert len(nb_states) == len(nb_actions)

        states = []
        actions = []

        

        for i in range(len(nb_states)):
            
            sample_states = user_histo.iloc[0:sep].sample(nb_states[i])
            sample_actions = user_histo.iloc[-(n-sep):].sample(nb_actions[i])

                
            sample_state = []
            sample_action = []
            for j in range(nb_states[i]):
                row = sample_states.iloc[j]
                # FORMAT STATE
                state = str(row.loc['item']) + '&' + str(row.loc['rating'])
                sample_state.append(state)
            
            for j in range(nb_actions[i]):
                row = sample_actions.iloc[j]
                # FORMAT ACTION
                action = str(row.loc['item']) + '&' + str(row.loc['rating'])
                sample_action.append(action)
            
            states.append(sample_state)
            actions.append(sample_action)
            
        return states, actions

    def gen_train_test(self, test_ratio, seed=None):
        print('gen_train start')
        n = len(self.histo)

        if seed is not None:
            random.Random(seed).shuffle(self.histo)
        else:
            random.shuffle(self.histo)

        self.train = self.histo[:int((test_ratio * n))]
        self.test = self.histo[int((test_ratio * n)):]
        self.user_train = [h.iloc[1,1] for h in self.train]
        print(self.user_train[0])
        self.user_test = [h.iloc[1,1] for h in self.test]

    def write_csv(self, filename, histo_to_write, delimiter=';', action_ratio=0.8, 
                  max_samp_by_user=5, max_state=100, max_action=50, nb_states=[], nb_actions=[]):
         with open(filename, mode='w') as file:
            f_writer = csv.writer(file, delimiter=delimiter)
            f_writer.writerow(['state', 'action_reward', 'n_state'])
            for user_histo in histo_to_write:
                states, actions = self.sample_histo(user_histo, action_ratio, 
                                                    max_samp_by_user, max_state, max_action, nb_states, nb_actions)
                for i in range(len(states)):
                    # FORMAT STATE
                    state_str = '|'.join(states[i])
                    # FORMAT ACTION
                    action_str = '|'.join(actions[i])
                    # FORMAT N_STATE
                    n_state_str = state_str + '|' + action_str
                    f_writer.writerow([state_str, action_str, n_state_str])

# Hyperparameters
history_length = 12 # N in article
ra_length = 4 # K in article
discount_factor = 0.99 # Gamma in Bellman equation
actor_lr = 0.0001
critic_lr = 0.001
tau = 0.001 # τ in Algorithm 3
batch_size = 16
nb_episodes = 100
nb_rounds = 50
filename_summary = 'summary.txt'
alpha = 0.5 # α (alpha) in Equation (1)
gamma = 0.9 # Γ (Gamma) in Equation (4)
buffer_size = 1000000 # Size of replay memory D in article
fixed_length = True # Fixed memory length
'''
data_path ='./book_up20.csv'

data = pd.read_csv(data_path, names={'item': int, 'user': str, 'rating': int, 'timestamp': int}, header=0)
print("789")
dg = DataGenerator(data_path)
dg.gen_train_test(0.8, seed=42)

print(len(dg.train))
print(len(dg.test))
print('train: ', dg.train[:1])
print('test:', dg.test[:1])
'''
