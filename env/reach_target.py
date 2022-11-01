from typing import List, Tuple, Union
import numpy as np
import torch as T
from pyrep.objects.shape import Shape
from pyrep.objects.proximity_sensor import ProximitySensor
from rlbench.const import colors
from rlbench.backend.task import Task
from rlbench.backend.spawn_boundary import SpawnBoundary
from rlbench.backend.conditions import DetectedCondition


class ReachTarget(Task):

    def init_task(self) -> None:
        self.target = Shape('target')
        self.distractor0 = Shape('distractor0')
        self.distractor1 = Shape('distractor1')
        self.boundaries = Shape('boundary')
        success_sensor = ProximitySensor('success')
       
        self.register_success_conditions(
            [DetectedCondition(self.robot.arm.get_tip(), success_sensor)])

    def init_episode(self, index: int) -> List[str]:
        color_name, color_rgb = colors[index]
        self.target.set_color(color_rgb)
        color_choices = np.random.choice(
            list(range(index)) + list(range(index + 1, len(colors))),
            size=2, replace=False)
        for ob, i in zip([self.distractor0, self.distractor1], color_choices):
            name, rgb = colors[i]
            ob.set_color(rgb)
        b = SpawnBoundary([self.boundaries])
        for ob in [self.target, self.distractor0, self.distractor1]:
            b.sample(ob, min_distance=0.2,
                     min_rotation=(0, 0, 0), max_rotation=(0, 0, 0))

        return ['reach the %s target' % color_name,
                'touch the %s ball with the panda gripper' % color_name,
                'reach the %s sphere' %color_name]

    def variation_count(self) -> int:
        return len(colors)

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]

    def get_low_dim_state(self) -> np.ndarray:
        # One of the few tasks that have a custom low_dim_state function.
        # return np.array(self.target.get_position()).flatten()
        desired_goal=self.sample_goal()
        observation=self.get_obs()
        return np.array({
            'observation': observation,
            'goal':desired_goal
            
        }).flatten()
        
    def get_obs(self):
        low_dim_data = [] 
        for data in [self.robot.arm.get_joint_velocities(), self.robot.arm.get_joint_positions(),
                     self.robot.arm.get_joint_forces(),
                     self.robot.gripper.get_pose(), self.robot.gripper.get_joint_positions(),
                     self.robot.gripper.get_joint_forces(),Shape('target').get_pose()]:
            if data is not None:
                low_dim_data.append(data)
        return np.concatenate(low_dim_data) if len(low_dim_data) > 0 else np.array([])

    def sample_goal(self):
        random_pose=self.get_obs()+np.random.uniform(-1,1,39)
        
        random_pose+=np.random.normal(scale=0.005,size=random_pose.shape)
        goal=self.get_achieved_goal()
        offset=random_pose-goal
        offset/=np.linalg.norm(offset)
        goal=goal-0.005*offset
        if np.random.uniform()<0.1:
            goal=self.get_achieved_goal().copy()
        return goal.flatten()


    def is_static_workspace(self) -> bool:
        return True
    def computing_disance(self,position1,position2):
        return np.linalg.norm(position1-position2,axis=-1)
    
    def get_achieved_goal(self):
        goal= self.get_obs()
        return goal

    def reward(self) -> Union[float, None]:
        #gripper touches object
        reward_ct=0
        target_obj_pose=self.target.get_position()
        gripper_pose=self.robot.gripper.get_position()
        reach_distance=self.computing_disance(target_obj_pose,gripper_pose)
        
        if round(reach_distance,2)<=0.10 :
            reward_ct=1
        else:
            reward_ct=0 
        
            
        return round(reward_ct,2)

    
    