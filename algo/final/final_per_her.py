import numpy as np
from collections import deque
class Buffer(object):
    def __init__(self,memory_size,input_dims,n_actions,batch_size):

        self.memory_size=memory_size
        self.counter=0
        self.batch_size=batch_size
         # initializes the state, next_state, action, reward, and terminal experience memory

        
        self.state_memory = np.zeros((self.memory_size, input_dims),dtype=np.float32)
        self.next_state_memory = np.zeros((self.memory_size, input_dims), dtype=np.float32)
        self.reward_memory = np.zeros(self.memory_size, dtype=np.float32)
        self.action_memory = np.zeros((self.memory_size, n_actions), dtype=np.float32)
        self.terminal_memory = np.zeros(self.memory_size, dtype=bool)
        # self.goal= np.zeros((self.memory_size, input_dims), dtype=np.float32)
       
    def store_transition(self, state, action, reward, next_state,done):
        
        
        curr_index = self.counter % self.memory_size
        
        state_array=np.array(list(state.values()),dtype=np.float32)
        next_state_array=np.array(list(next_state.values()),dtype=np.float32)
       
        state=state_array.flatten()
        next_state=next_state_array.flatten()

        self.state_memory[curr_index] = state
        self.action_memory[curr_index] = action
        self.reward_memory[curr_index] = reward
        self.next_state_memory[curr_index] = next_state
        self.terminal_memory[curr_index] = done
        # self.goal[curr_index]=goal
        

        self.counter += 1

    def sample_buffer(self):
        
        rand_index = np.random.choice(min(self.counter, self.memory_size), self.batch_size)

        rand_state = self.state_memory[rand_index]
        rand_action = self.action_memory[rand_index]
        rand_reward = self.reward_memory[rand_index]
        rand_next_state = self.next_state_memory[rand_index]
        rand_done = self.terminal_memory[rand_index]
        # rand_goal=self.goal[rand_index]
       

        return rand_state, rand_action, rand_reward, rand_next_state,rand_done

class PrioritizedReplay(object):
    """
    Proportional Prioritization
    """
    def __init__(self, capacity,input_dims, n_actions,batch_size, alpha=0.6, beta_start = 0.4, beta_frames=int(1e5)):
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1 #for beta calculation
        self.capacity   = capacity
        self.buffer     = deque(maxlen=capacity)
        self.pos        = 0
        self.priorities = deque(maxlen=capacity)
        self.counter=0
        self.normal_buffer=Buffer(capacity,input_dims,n_actions,batch_size)
    
    def beta_by_frame(self, frame_idx):
          return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)
    
    def store_transition(self, state, action, reward, next_state, done):
        
       
        state_array=np.array(list(state.values()),dtype=np.float32)
        next_state_array=np.array(list(next_state.values()),dtype=np.float32)
       
        state=state_array.flatten()
        next_state=next_state_array.flatten()
        
        max_prio = max(self.priorities) if self.buffer else 1.0 # gives max priority if buffer is not empty else 1
        
        self.buffer.insert(0, (state, action, reward, next_state, done))
        self.priorities.insert(0, max_prio)
        self.counter+=1
    
    
    def sample(self, batch_size, c_k=2500):
        N = len(self.buffer)
        if c_k > N:
            c_k = N
            

        if N == self.capacity:
            prios = np.array(self.priorities)
        else:
            prios = np.array(list(self.priorities)[:c_k])
        
        #(prios)
        # calc P = p^a/sum(p^a)
        probs  = prios ** self.alpha
        P = probs/probs.sum()
      
        indices = np.random.choice(c_k, batch_size, p=P) 
        samples = [self.buffer[idx] for idx in indices]
        
        beta = self.beta_by_frame(self.frame)
        self.frame+=1
     
        weights  = (c_k * P[indices]) ** (-beta)
        # normalize weights
        weights /= weights.max() 
        weights  = np.array(weights, dtype=np.float32) 
        
        states, actions, rewards, next_states, dones = zip(*samples) 
        return np.concatenate(states), actions, rewards, np.concatenate(next_states), dones, indices, weights
    
    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            idx=int(idx)
            self.priorities[idx] = int(prio)

    def __len__(self):
        return len(self.buffer)
