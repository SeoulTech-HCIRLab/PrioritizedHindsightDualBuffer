import os
import torch as T
import torch.nn.functional as F
import numpy as np
from final_per_her import Buffer,PrioritizedReplay
from sac import ActorNetwork, CriticNetwork, ValueNetwork


class perAgent():
    def __init__(self, input_dims,n_actions,action_space,flag,alpha=0.0001, beta=0.0003,
            env=None,env_params=None,gamma=0.99, max_size=10000000, tau=0.005,
            layer1_size=256, layer2_size=256, batch_size=64, reward_scale=8):
        self.gamma = gamma
        self.tau = tau
        self.env=env
        self.env_params=env_params
        device=T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.prio_memory = PrioritizedReplay(max_size,input_dims,n_actions,batch_size)
        self.regular_memory=Buffer(max_size,input_dims,n_actions,batch_size)
    
        self.batch_size = batch_size
        self.max_size=max_size
        self.n_actions = n_actions
  
        

        self.actor = ActorNetwork(alpha, input_dims, n_actions=n_actions,
                    name='actor', max_action=action_space.high,flag=flag)
        self.critic_1 = CriticNetwork(beta, input_dims, n_actions=n_actions,
                    name='critic_1',flag=flag)
        self.critic_2 = CriticNetwork(beta, input_dims, n_actions=n_actions,
                    name='critic_2',flag=flag)
        self.value = ValueNetwork(beta, input_dims,flag, name='value')
        self.target_value = ValueNetwork(beta, input_dims,flag, name='target_value')
        self.scale = reward_scale
        self.update_network_parameters(tau=1)
    

    def choose_goal_action(self, observation,goal):
        state =T.Tensor([np.concatenate([observation,goal])]).to(self.actor.device)
        actions, _ = self.actor.sample_normal(state, flag=True,reparameterize=False)
        return actions.cpu().detach().numpy()[0]
 

    def remember(self, state, action, reward, new_state, done):
        self.prio_memory.store_transition(state, action, reward, new_state, done)

    # def replayremember(self,state,action,reward,new_state,done,goal):
    #     self.regular_memory.store_transition(state, action, reward, new_state, done,goal)

   
    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        target_value_params = self.target_value.named_parameters()
        value_params = self.value.named_parameters()

        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        for name in value_state_dict:
            value_state_dict[name] = tau*value_state_dict[name].clone() + \
                    (1-tau)*target_value_state_dict[name].clone()

        self.target_value.load_state_dict(value_state_dict)

    def save_models(self):
        print('.... saving models ....')
        self.actor.save_checkpoint()
        self.value.save_checkpoint()
        self.target_value.save_checkpoint()
        self.critic_1.save_checkpoint()
        self.critic_2.save_checkpoint()

    def load_models(self):
        print('.... loading models ....')
        self.actor.load_checkpoint()
        self.value.load_checkpoint()
        self.target_value.load_checkpoint()
        self.critic_1.load_checkpoint()
        self.critic_2.load_checkpoint()
    # def step(self):
    #     experience=self.memory.sample_()
    #     self.learn(experience)

    def learn(self,notproposed=False,checkprop=True):
        buffer=['priority','regular']
        n=np.random.choice(['priority','regular'],1,p=[0.3,0.7])
        
        
        if self.prio_memory.counter < 200:
            return
    
      
        if notproposed:
            state, action, reward,state_,done,weights,indices=self.replaymemory.sample_buffer(self.batch_size)
            reward = T.tensor(reward, dtype=T.float).to(self.critic_1.device)
            done = T.tensor(done).to(self.critic_1.device)
            state_ = T.tensor(state_, dtype=T.float).to(self.critic_1.device)
            state = T.tensor(state, dtype=T.float).to(self.critic_1.device)
            action = T.tensor(action, dtype=T.float).to(self.critic_1.device)
            value = self.value(state).view(-1)
            value_ = self.target_value(state_).view(-1)
            value_[done] = 0.0

            actions, log_probs = self.actor.sample_normal(state,flag=False,reparameterize=False)
            log_probs = log_probs.view(-1)
            q1_new_policy = self.critic_1.forward(state, actions)
            q2_new_policy = self.critic_2.forward(state, actions)
            critic_value = T.min(q1_new_policy, q2_new_policy)
            critic_value = critic_value.view(-1)

            self.value.optimizer.zero_grad()
            value_target = critic_value - log_probs
            value_loss = 0.5 * F.mse_loss(value, value_target)
            value_loss.backward(retain_graph=True)
            self.value.optimizer.step()

            actions, log_probs = self.actor.sample_normal(state,flag=False, reparameterize=True)
            log_probs = log_probs.view(-1)
            q1_new_policy = self.critic_1.forward(state, actions)
            q2_new_policy = self.critic_2.forward(state, actions)
            critic_value = T.min(q1_new_policy, q2_new_policy)
            critic_value = critic_value.view(-1)
        
            actor_loss = (log_probs - critic_value)
            
            actor_loss = T.mean(actor_loss)
            self.actor.optimizer.zero_grad()
         
            actor_loss.backward(retain_graph=True)

            self.critic_1.optimizer.zero_grad()
            self.critic_2.optimizer.zero_grad()
            reward=reward.reshape(-1)

            q_hat = self.scale*reward + self.gamma*value_

            q1_old_policy = self.critic_1.forward(state, action).view(-1)
            q2_old_policy = self.critic_2.forward(state, action).view(-1)
            critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
            critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)
          
        
        

       
            #critic 1
            critic_t_loss=critic_1_loss+critic_2_loss
            critic_t_loss.backward()
            self.critic_1.optimizer.step()
            self.critic_2.optimizer.step()
            # critic 2
            
        

        
       
            self.update_network_parameters()
             


        if checkprop:
            state, action, reward,state_,done,weights,indices=[],[],[],[],[],[],[]
            
            if(buffer[1]==n):
                print('reg')
                
                if(self.regular_memory.counter>200):
                    state, action, reward,state_,done = self.regular_memory.sample_buffer(self.batch_size)
                   
                     
                else:
                    n=buffer[0]
                
            if(buffer[0]==n):
                print('pri')
                state, action, reward,state_,done,weights, indices = self.prio_memory.sample(self.batch_size)
                
                  

            
            reward = T.tensor(reward, dtype=T.float).to(self.critic_1.device)
            done = T.tensor(done).to(self.critic_1.device)
            state_ = T.tensor(state_, dtype=T.float).to(self.critic_1.device).reshape(64,-1)
            state = T.tensor(state, dtype=T.float).to(self.critic_1.device).reshape(64,-1)
            action = T.tensor(action, dtype=T.float).to(self.critic_1.device).reshape(64,-1)
            weights=T.FloatTensor(weights).to(self.critic_1.device)
           
        
           

            value = self.value(state).view(-1)
            value_ = self.target_value(state_).view(-1)
            value_[done] = 0.0
            actions, log_probs = self.actor.sample_normal(state,flag=False,reparameterize=True)
            log_probs = log_probs.view(-1)
            q1_new_policy = self.critic_1.forward(state, actions)
            q2_new_policy = self.critic_2.forward(state, actions)
            critic_value = T.min(q1_new_policy, q2_new_policy)
            critic_value = critic_value.view(-1)
            

            self.value.optimizer.zero_grad()
            value_target = critic_value - log_probs
            value_loss = 0.5 * F.mse_loss(value, value_target)
            value_loss.backward(retain_graph=True)
            self.value.optimizer.step()

            actions, log_probs = self.actor.sample_normal(state,flag=False, reparameterize=True)
            log_probs = log_probs.view(-1)
            q1_new_policy = self.critic_1.forward(state, actions)
            q2_new_policy = self.critic_2.forward(state, actions)
            critic_value = T.min(q1_new_policy, q2_new_policy)
            critic_value = critic_value.view(-1)
        
            actor_loss = (log_probs - critic_value)
            if len(weights):
                actor_loss=actor_loss*weights
            actor_loss = T.mean(actor_loss)
            self.actor.optimizer.zero_grad()
         
            actor_loss.backward(retain_graph=True)

            self.critic_1.optimizer.zero_grad()
            self.critic_2.optimizer.zero_grad()
            reward=reward.reshape(-1)

            q_hat = self.scale*reward + self.gamma*value_

            q1_old_policy = self.critic_1.forward(state, action).view(-1)
            q2_old_policy = self.critic_2.forward(state, action).view(-1)
            critic_1_loss = 0.5 * F.mse_loss(q1_old_policy, q_hat)
            critic_2_loss = 0.5 * F.mse_loss(q2_old_policy, q_hat)
            if len(weights):
                td_error1=q1_old_policy-q_hat
                td_error2=q2_old_policy-q_hat
                td_sum=(td_error1+td_error2).detach().cpu().numpy()
                prios=abs(((td_sum)/2.0 + 1e-5).squeeze())
        
        

       
            #critic 1
            critic_t_loss=critic_1_loss+critic_2_loss
            critic_t_loss.backward()
            self.critic_1.optimizer.step()
            self.critic_2.optimizer.step()
            # critic 2
       
       
        
            if len(indices)   :
                self.prio_memory.update_priorities(indices,prios)

        
       
            self.update_network_parameters()

