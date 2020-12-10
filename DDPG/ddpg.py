import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
from sklearn.model_selection import train_test_split
import tensorflow as tf
from torch.utils.data import DataLoader
import itertools
import torch

import os 

from collections import defaultdict
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchtools.optim import Ranger
from ranger import Ranger

import tqdm
import random
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity as cs
import seaborn as sns

datapath = os.getenv('HOME') + '/project/thebest_52/ratings_hot.csv'
#columns =  ['UserID', 'BookID', 'Rating', 'Timestamp']
R_df = pd.read_csv(datapath, sep = ",", dtype = int)


userids = list(R_df.index.values) #list of userids
idx_to_userids = {i:userids[i] for i in range(len(userids))}
userids_to_idx = {userids[i]:i for i in range(len(userids))}

print(R_df.head())