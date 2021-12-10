import numpy as np

import gym
import metaworld
from . import register_env


@register_env('ml1-push')
class MWML1PushEnv(gym.Env):
    """Class to wrap the meta-world ML1 Push environment.

    [1] Tianhe Yu, Deirdre Quillen, Zhanpeng He, Ryan Julian, Karol Hausman, Chelsea Finn, 
        Sergey Levine, "Meta-World: A Benchmark and Evaluation for Multi-Task and 
        Meta Reinforcement Learning", 2019
        (https://arxiv.org/abs/1910.10897)
    """
    def __init__(self, task={}, n_tasks=100, n_train_tasks=50, n_eval_tasks=50, randomize_tasks=True):
        super(MWML1PushEnv, self).__init__()
        self._ml1 = metaworld.ML1('push-v2')
        self._env = self._ml1.train_classes['push-v2']()
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self.sim = self._env.sim

        #self.tasks = self.sample_tasks(n_tasks)
        self.tasks = self.sample_tasks(n_train_tasks, n_eval_tasks)
        self._task = self.tasks[0]
        self._goal = self.tasks[0]['goal']

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        return (obs.astype(np.float32), reward, done, info)

    def reset(self):
        obs = self._env.reset()
        return obs.astype(np.float32)

    def sample_tasks(self, num_tasks):
        goals = self._env.train_tasks + self._env.test_tasks
        tasks = [{'goal': goal} for goal in goals]
        return tasks

    def sample_tasks(self, num_train, num_eval, randomize_tasks=True):
        train_idxs = np.arange(len(self._ml1.train_tasks))
        test_idxs = np.arange(len(self._ml1.test_tasks))
        if randomize_tasks:
            np.random.shuffle(train_idxs)
            np.random.shuffle(test_idxs)
        train_idxs = train_idxs[ : num_train]
        test_idxs = test_idxs[ : num_eval]

        sampled_train = [self._ml1.train_tasks[i] for i in train_idxs]
        sampled_test = [self._ml1.test_tasks[i] for i in test_idxs]
        goals = sampled_train + sampled_test
        tasks = [{'goal': goal} for goal in goals]
        return tasks

    def get_all_task_idx(self):
        return range(len(self.tasks))

    def reset_task(self, idx):
        self._task = self.tasks[idx]
        self._goal = self._task['goal']
        self._env.set_task(self._goal)
        self._env.reset()
