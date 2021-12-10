import gym
import numpy as np
import metaworld

class ML45Env(gym.Env):
    def __init__(self):
        super(ML45Env, self).__init__()
        self.ml45 = metaworld.ML45()
        self._env = self.ml45.train_classes['push-v2']()
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
        all_envs = self.ml45.train_classes if train else self.ml45.test_classes
        all_tasks = self.ml45.train_tasks if train else self.ml45.test_tasks
        if num_tasks % len(all_envs) != 0:
            raise ValueError('`num_tasks` should be in mutliples of number of benchmarks')
        tasks_per_env = num_tasks // len(all_envs)

        tasks = []
        for benchmark_name, env in all_envs.items():
            env_tasks_idx = []
            for idx, task in enumerate(all_tasks):
                if task.env_name == benchmark_name:
                    env_tasks_idx.append(idx)
            env_tasks_idx = np.array(env_tasks_idx)
            idxs = np.random.randint(low=0, high=len(env_tasks_idx), size=(tasks_per_env,))
            sampled_tasks_idx = env_tasks_idx[idxs]
            for i in sampled_tasks_idx:
                tasks.append({'benchmark_name': benchmark_name, 'task_idx': i, 'train':train})
        # shuffle tasks
        tasks = np.array(tasks)
        np.random.shuffle(tasks)
        return tasks.tolist()

    def reset_task(self, task):
        benchmark_name = task['benchmark_name']
        task_idx = task['task_idx']
        if task['train']:
            self._env = self.ml45.train_classes[benchmark_name]()
            self._env.set_task(self.ml45.train_tasks[task_idx])
        else:
            self._env = self.ml45.test_classes[benchmark_name]()
            self._env.set_task(self.ml45.test_tasks[task_idx])
        self._task = task

    def seed(self, value=None):
        return self._env.seed(value)
