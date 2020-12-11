# Offline Environment

class OfflineEnv:

    def __init__(self,dataloader,users_dict):
        self.dataloader = iter(dataloader)
        self.users_dict = users_dict
        self.data = next(self.dataloader) # {'item':items,'rating':ratings,'size':size,'userid':user_id,'idx':idx}
        self.memory = [item[0] for itmem in self.data['item']]
        self.user_history = self.users_dict[self.data['userid']]
        self.done = 0
        self.related_movie = self.generate_related_movie(self.user_history)
    
    def generate_related_movie(self,user_history):




    

    def reset(self):
        

    

    def step(self):
            ### Env : step
            rate = int(users_dict[int(userid_b[0])]["rating"][action])
            try:
                ratings = (int(rate)-3)/2
            except:
                ratings = 0
            reward = torch.Tensor((ratings,))
            ep_reward = ep_reward + ratings

            if reward > 0:
                update_memory(memory,int(users_dict[int(userid_b[0])]["item"][action]), idx_b)

            next_state = drrave_state_rep(userid_b,item_b,memory,idx_b)
            next_state_rep = torch.reshape(next_state,[-1])

        return state, reward, done