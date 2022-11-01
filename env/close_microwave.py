from typing import List, Tuple,Union
import numpy as np
from pyrep.objects.shape import Shape
from pyrep.objects.joint import Joint
from pyrep.objects.object import Object
from rlbench.backend.task import Task
from rlbench.backend.conditions import JointCondition



class CloseMicrowave(Task):

    def init_task(self) -> None:
        self.register_success_conditions([JointCondition(
            Joint('microwave_door_joint'), np.deg2rad(40))])

    def init_episode(self, index: int) -> List[str]:
        return ['close microwave',
                'shut the microwave',
                'close the microwave door',
                'push the microwave door shut']

    def variation_count(self) -> int:
        return 1

    def base_rotation_bounds(self) -> Tuple[List[float], List[float]]:
        return [0, 0, -3.14 / 4.], [0, 0, 3.14 / 4.]

    def boundary_root(self) -> Object:
        return Shape('boundary_root')
    def get_low_dim_state(self) -> np.ndarray:
        # One of the few tasks that have a custom low_dim_state function.
        # return np.array(self.target.get_position()).flatten()
        # achieved_goal=self.get_achieved_goal()
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
                     self.robot.gripper.get_joint_forces(),Joint('microwave_door_joint').get_pose()]:
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


    
    def computing_disance(self,position1,position2):
        return np.linalg.norm(position1-position2,axis=-1)
    
    def get_achieved_goal(self):
        goal= self.get_obs()
        return goal
    
