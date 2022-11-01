# Prioritized Hindsight with Dual Buffer for meta-reinforcement learning
This repository contains the implementation of prioritized hindsight with dual buffer created by Sofanit Wubeshet Beyene and Ji-Hyeong Han from Seoul National University of Science and Technology.

## Abstract

Sharing prior knowledge across multiple robotic manipulation tasks is a challenging 
research topic. Although the state-of-the-art deep reinforcement learning (DRL) algorithms have 
shown immense success in single robotic tasks, it is still challenging to extend these algorithms to 
apply directly to resolve multi-task manipulation problems. This is mostly due to the problems 
associated with efficient exploration in high-dimensional state and continuous action spaces. Furthermore, in multi-task scenarios, the problem of sparse reward and sample inefficiency of DRL 
algorithms is exacerbated. Therefore, we propose a method to increase the sample efficiency of the 
soft-actor critic (SAC) algorithm and extend it to a multi-task setting. The agent learns a prior policy 
from two structurally similar tasks and adapts the policy to a target task.   
We propose a prioritized hindsight with dual experience replay to improve data storage and sampling technique which, in 
turn, assists the agent in performing structured exploration that leads to sample efficiency. The 
proposed method separate the experience replay buffer into two buffers to contain real trajectories 
and hindsight trajectories to reduce the bias introduced by the hindsight trajectories in the buffer.
Moreover, we utilize high-reward transitions from previous tasks to assist the network in easily 
adapting to the new task. We demonstrate the proposed method based on several manipulation tasks
using a 7-DoF robotic arm in RLBench. The experimental results show that the proposed method 
outperforms vanilla SAC in both single-task setting and multi-task setting.














<p float=left>
<img alt="p_reach_2" src="https://user-images.githubusercontent.com/33028604/199187825-97a6507b-8a19-4d33-b330-14f0ed1f4416.png" width="200" height="200"/>
<img alt= "closebox" src="https://user-images.githubusercontent.com/33028604/199187843-7b0eede7-7cbb-4171-90ac-91abcd523e71.png" width="200" height="200"/>
<img alt="closemicrowave" src="https://user-images.githubusercontent.com/33028604/199187864-0e628b05-6904-4d93-8d04-9cb94925e9c5.png" width="200" height="200"/>
</p>



## Installation
### Requirements
Environment  
  &nbsp;Python3.8  
  &nbsp;torch1.9.1  

Install RLBench  
>https://github.com/stepjam/RLBench
  
## How to run
### Replace environment files
~~~
env/reach_target.py by tasks/reach_target.py
env/close_box.py by tasks/close_box.py
env/close_mircowave.py to tasks/close_mircrowave.py
~~~
### Run  
~~~
python final_main.py 
~~~



