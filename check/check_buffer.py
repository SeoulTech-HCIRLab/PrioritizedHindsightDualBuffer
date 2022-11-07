import numpy as np
class Buffer:
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
