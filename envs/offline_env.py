# Offline Environment

class OfflineEnv:

    def __init__(self,dataloader,users_dict):
        self.dataloader = iter(dataloader)
        self.users_dict = users_dict
        self.data = next(self.dataloader) # {'item':items,'rating':ratings,'size':size,'userid':user_id,'idx':idx}
        self.memory = [item[0] for item in self.data['item']]
        self.user_history = self.users_dict[self.data['userid'][0]]
        self.done = 0
        self.related_books = self.generate_related_movie(self.user_history)
        self.viewed_pos_books = []
    
    def generate_related_movie(self,user_history):




    

    def reset(self):
        

    



    
    def update_memory(self,action):
        self.memory = list(self.memory[1:])+[action]
    
    def step(self,action):
            ### Env : step
            ### 
            rating = int(self.user_history["rating"][action])
            normal_rating = (int(rating)-3)/2
            reward = torch.Tensor((normal_rating,))
            #ep_reward = ep_reward + ratings

            if reward > 0:
                update_memory(action)
                self.viewed_pos_books.append(action)
        
            if len(self.viewed_pos_books) == len(self.related_books):
                done = 1
            
        return reward, done, self.memory


        #next_state = drrave_state_rep(userid_b,item_b,memory,idx_b)
        #next_state_rep = torch.reshape(next_state,[-1])
