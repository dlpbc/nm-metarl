# -*- coding: utf-8 -*-
import numpy as np
import gym
from gym.spaces import Box
from mcgridenv import MazeDepth2v1Grid

class MCEnvGoalChange(MazeDepth2v1Grid):
    def __init__(self):
        super(MCEnvGoalChange, self).__init__()
        self._task = {}
        self._goal = None
        self._state = None
        # override observation shape, flatten to 1d vector
        shape = (np.prod(self.observation_space.shape), )
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=shape)
        
    def set_config(self, env_config_path):
        return

    def seed(self, seed=None):
        if seed is not None:
            super(MCEnvGoalChange, self).seed(seed)

    def sample_tasks(self, num_tasks):
        goals = []
        for _ in range(num_tasks):
            goals.append(self.sample_goal())
        tasks = [{'goal': goal} for goal in goals]
        return tasks

    def reset_task(self, task):
        self._task = task
        self._goal = task['goal']
        self.set_goal(self._goal)

    def reset(self, env=True):
        obs, _, _, _ = super(MCEnvGoalChange, self).reset()
        obs = obs.astype(np.float32) / 255.
        obs = obs.ravel()
        self._state = obs
        return self._state

    def step(self, action):
        obs, reward, done, info = super(MCEnvGoalChange, self).step(action)
        obs = obs.astype(np.float32) / 255.
        obs = obs.ravel()
        return obs, reward, done, info
