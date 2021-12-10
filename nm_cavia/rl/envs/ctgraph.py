# -*- coding: utf-8 -*-
import numpy as np
import gym
from gym import spaces
from gym_CTgraph import CTgraph_env
from gym_CTgraph.CTgraph_env import CTgraphEnv
from gym_CTgraph.CTgraph_plot import CTgraph_plot
from gym_CTgraph.CTgraph_conf import CTgraph_conf
from gym_CTgraph.CTgraph_images import CTgraph_images
from itertools import product

class CTgraphEnvGoalChange(CTgraphEnv):
    def __init__(self):
        super(CTgraphEnvGoalChange, self).__init__()
        
    def set_config(self, env_config_path):
        configuration = CTgraph_conf(env_config_path)
        conf_data = configuration.getParameters()
        imageDataset = CTgraph_images(conf_data)
        self.init(conf_data, imageDataset)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=self._state.shape, dtype=np.float32)

    def init(self, conf_data, images):
        depth = conf_data['graph_shape']['depth']
        branch = conf_data['graph_shape']['branching_factor']
        self.all_tasks = product(list(range(branch)), repeat=depth) 
        self.all_tasks = [{'goal': g} for g in self.all_tasks]

        obs, _, _, _ = super(CTgraphEnvGoalChange, self).init(conf_data, images)
        self._goal = self.get_high_reward_path()
        self._task = {'goal': self._goal}
        obs = obs.astype(np.float32) / 255.
        obs = obs.ravel()
        self._state = obs

    def seed(self, seed=None):
        # ignore seed param and use CT-graph seed
        # that is already set
        return None

    def sample_tasks(self, num_tasks):
        #goals = []
        #for _ in range(num_tasks):
        #    goals.append(self.get_random_path())
        #tasks = [{'goal': goal} for goal in goals]
        idxs = np.random.randint(low=0, high=len(self.all_tasks), size=num_tasks)
        tasks = [self.all_tasks[idx] for idx in idxs]
        return tasks

    def reset_task(self, task):
        self._task = task
        self._goal = task['goal']
        self.set_high_reward_path(self._goal)

    def reset(self, env=True):
        obs, _, _, _ = super(CTgraphEnvGoalChange, self).reset()
        obs = obs.astype(np.float32) / 255.
        obs = obs.ravel()
        self._state = obs
        return self._state

    def step(self, action):
        obs, reward, done, info = super(CTgraphEnvGoalChange, self).step(action)
        obs = obs.astype(np.float32) / 255.
        obs = obs.ravel()
        return obs, reward, done, {'info': info}
