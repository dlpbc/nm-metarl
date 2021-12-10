import gym
import numpy as np
import metaworld

class ML1Env(gym.Env):
    def __init__(self, benchmark_name):
        super(ML1Env, self).__init__()
        self.ml1 = metaworld.ML1(benchmark_name)
        self._env = self.ml1.train_classes[benchmark_name]()
        self._task = None
        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space
        self.seed()

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        return obs.astype(np.float32), reward, done, info

    def reset(self):
        obs = self._env.reset()
        return obs.astype(np.float32)

    def sample_tasks(self, num_tasks, train=True):
        all_tasks = self.ml1.train_tasks if train else self.ml1.test_tasks
        idxs = np.random.randint(low=0, high=len(all_tasks), size=(num_tasks,))
        tasks = [{'task':all_tasks[idx]} for idx in idxs]
        return tasks

    def reset_task(self, task):
        self._env.set_task(task['task'])
        self._task = task

    def seed(self, value=None):
        return self._env.seed(value)
