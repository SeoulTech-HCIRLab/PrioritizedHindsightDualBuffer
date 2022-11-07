'''HER MULTI TASK'''


import gym
import rlbench.gym
import numpy as np
from agent import Agent
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from rlbench.environment import Environment
from rlbench.action_modes import ArmActionMode, ActionMode
from rlbench.observation_config import ObservationConfig
from rlbench.tasks import MT30_V1
from gym import spaces
import torch.nn as nn
import torch as T
from utils import plotLearning
from check_multi_agent import perAgent



def compute_reward(next_state,goal_state):
    
    state=T.tensor(next_state)

    goal=T.tensor(goal_state)
    done=T.equal(state,goal)
    
    
    return 1 if done else 0 ,done


if __name__ == '__main__':
    obs_config = ObservationConfig()
    obs_config.set_all(True)
    action_mode = ActionMode(ArmActionMode.ABS_JOINT_VELOCITY)
    env = Environment(
    action_mode, obs_config=obs_config, headless=True)
    train_tasks = MT30_V1['train'][:2]
    write=SummaryWriter()
    training_cycles_per_task=1
    training_steps_per_task=300
    episode_length=200
    future_k=4
   
    
    load_checkpoint = False
    ave_rew=[]
    for i in range(training_cycles_per_task):
        train=np.random.choice(train_tasks,1)[0]
        
        
        task=env.get_task(train)
        eps_trajectory=[]
        total_score=0   
        _,obs=task.reset()
        

        for k in range(episode_length):
            score=0
            
            
            
            for step in range(training_steps_per_task):
                terminate=False
                action_space = spaces.Box(low=-1.0, high=1.0, shape=(env.action_size,))
                observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs.task_low_dim_state[0]['observation'].shape)
                agent = perAgent(input_dims=2*observation_space.shape[0], env=task,n_actions=action_space.shape[0],action_space=action_space,flag=False)
                action=agent.choose_goal_action(obs.task_low_dim_state[0]['observation'],obs.task_low_dim_state[0]['goal'])
                obs_,reward,done=task.step(action)
                eps_trajectory.append([obs.task_low_dim_state[0],action,reward,obs_.task_low_dim_state[0],done])
                obs=obs_
                if done:
                    success+=reward                     
                    break
                       
                steps_taken=step
                for t in range(steps_taken):
                    state,action,reward_,next_state,done=eps_trajectory[t]
                    #normal replay
                    agent.remember(state,action,reward_,next_state,done)

                    for _ in range(future_k):

                     
                        #hingsight replay

                        future=np.random.randint(t,steps_taken)
                        eps_traj=eps_trajectory[t]
                        new_goal=eps_trajectory[future][3]['observation']
                        nw_rw,nw_done=compute_reward(next_state['observation'],new_goal)
                        
                        eps_traj[0]['goal']=new_goal
                        eps_traj[3]['goal']=new_goal
                        st=eps_traj[0]
                        nxt_state=eps_traj[3]
 
                        agent.remember(st,action,nw_rw,nxt_state,nw_done)
                          
            
                
                if not load_checkpoint:
                    agent.learn()
              
            ave_rew.append(score)            
                           
            
            print('task ', task.get_name(), 'episode', k, 'score %.1f' % score)
            write.add_scalar("success_graph",success,score)
            write.add_scalar("Episodic_reward",ave_rew,k) 
        
       
            
        
            
           
     

      
        
