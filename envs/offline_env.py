# Offline Environment
import torch
import numpy as np

class OfflineEnv:

    def __init__(self, dataloader, users_dict, item_embeddings_dict):
        
        self.dataloader = iter(dataloader)
        self.users_dict = users_dict
        
        self.data = next(self.dataloader) # {'item':items,'rating':ratings,'size':size,'userid':user_id,'idx':idx}
        self.user_history = self.users_dict[self.data['userid'][0]]

        self.item_embedding = torch.Tensor([np.array(item_embeddings_dict[item]) for item in users_dict[int(self.data['userid'][0])]['item']])
        self.items = self.item_embedding.T.unsqueeze(0)

        self.memory = [item[0] for item in self.data['item']]
        self.done = 0

        self.related_books = self.generate_related_books()
        self.viewed_pos_books = []
    
    def generate_related_books(self):
        related_movie = []
        items = self.user_history['items'][10:]
        ratings = self.user_history['ratings'][10:]

        for item, rating in zip(items, ratings):
            if rating > 3:
                related_movie.append(item)
        
        return related_movie

    def reset(self):
        self.data = next(self.dataloader)
        self.memory = [item[0] for item in self.data['item']]
        self.user_history = self.users_dict[self.data['userid'][0]]
        self.done = 0
        self.related_books = self.generate_related_books()
        self.viewed_pos_books = []

    def update_memory(self,action):
        self.memory = list(self.memory[1:])+[action]
    
    def step(self, action):
        
        ### Env : step
        ### 
        rating = int(self.user_history["rating"][action])
        normal_rating = (int(rating)-3)/2
        reward = torch.Tensor((normal_rating,))
        #ep_reward = ep_reward + ratings

        if reward > 0:
            self.update_memory(action)
            self.viewed_pos_books.append(action)
    
        if len(self.viewed_pos_books) == len(self.related_books):
            self.done = 1
        
        return self.memory, reward, self.done


        #next_state = drrave_state_rep(userid_b,item_b,memory,idx_b)
        #next_state_rep = torch.reshape(next_state,[-1])