import numpy as np

import gym
import metaworld
from . import register_env


@register_env('ml45')
class MWML45Env(gym.Env):
    """Class to wrap the meta-world ML45 benhmarks.

    [1] Tianhe Yu, Deirdre Quillen, Zhanpeng He, Ryan Julian, Karol Hausman, Chelsea Finn, 
        Sergey Levine, "Meta-World: A Benchmark and Evaluation for Multi-Task and 
        Meta Reinforcement Learning", 2019
        (https://arxiv.org/abs/1910.10897)
    """
    def __init__(self, task={}, n_tasks=250, n_train_tasks=225, n_eval_tasks=25, randomize_tasks=True):
        super(MWML45Env, self).__init__()
        self._ml45 = metaworld.ML45()
        self._env = self._ml45.train_classes['push-v2']()
        self._goal = None
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self.sim = self._env.sim

        #self.tasks = self.sample_tasks(n_tasks)
        self.tasks = self.sample_tasks(n_train_tasks, n_eval_tasks)
        self._task = self.tasks[0]

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        return (obs.astype(np.float32), reward, done, info)

    def reset(self):
        obs = self._env.reset()
        return obs.astype(np.float32)

    def sample_tasks(self, num_tasks):
        raise NotImplementedError

    def sample_tasks(self, num_train, num_eval, randomize_tasks=True):
        def _sampler(all_envs, all_tasks, num_tasks, shuffle, train=True):
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
            if shuffle:
                tasks = np.array(tasks)
                np.random.shuffle(tasks)
                tasks = tasks.tolist()
            return tasks

        all_envs_train = self._ml45.train_classes
        all_tasks_train = self._ml45.train_tasks
        all_envs_eval = self._ml45.test_classes
        all_tasks_eval = self._ml45.test_tasks
        tasks_train = _sampler(all_envs_train, all_tasks_train, num_train, randomize_tasks, True)
        tasks_eval = _sampler(all_envs_eval, all_tasks_eval, num_eval, randomize_tasks, False)
        return tasks_train + tasks_eval

    def get_all_task_idx(self):
        return range(len(self.tasks))

    def reset_task(self, idx):
        self._task = self.tasks[idx]
        benchmark_name = self._task['benchmark_name']
        task_idx = self._task['task_idx']
        train = self._task['train']
        self._goal = benchmark_name + str(task_idx) + str(train)

        if train:
            self._env = self._ml45.train_classes[benchmark_name]()
            self._env.set_task(self._ml45.train_tasks[task_idx])
        else:
            self._env = self._ml45.test_classes[benchmark_name]()
            self._env.set_task(self._ml45.test_tasks[task_idx])
        self._env.reset()
