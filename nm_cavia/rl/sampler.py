import numpy as np
import multiprocessing as mp
from inspect import signature

import gym
import torch

from envs.subproc_vec_env import SubprocVecEnv
#from episode import BatchEpisodes
from episode import BatchEpisodes, BatchEpisodesAnalysis


def make_env(env_name, env_config_path=None):
    def _make_env():
        env = gym.make(env_name)
        if env_config_path is not None:
            env.set_config(env_config_path)
        return env

    return _make_env


class BatchSampler(object):
    def __init__(self, env_name, batch_size, device, seed, num_workers=mp.cpu_count() - 1,
                env_config_path=None):
        self.env_name = env_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device

        self.queue = mp.Queue()
        self.envs = SubprocVecEnv([make_env(env_name, env_config_path) for _ in range(num_workers)],\
                    queue=self.queue)
        self.envs.seed(seed)
        self._env = gym.make(env_name)
        if env_config_path is not None:
            print(env_config_path)
            self._env.set_config(env_config_path)
        self._env.seed(seed)

    def sample(self, policy, params=None, gamma=0.95, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        episodes = BatchEpisodes(batch_size=batch_size, gamma=gamma, device=self.device)
        for i in range(batch_size):
            self.queue.put(i)
        for _ in range(self.num_workers):
            self.queue.put(None)
        observations, batch_ids = self.envs.reset()
        dones = [False]
        while (not all(dones)) or (not self.queue.empty()):
            with torch.no_grad():
                observations_tensor = torch.from_numpy(observations).to(device=self.device)
                actions_tensor = policy(observations_tensor, params=params).sample()
                actions = actions_tensor.cpu().numpy()
            new_observations, rewards, dones, new_batch_ids, _ = self.envs.step(actions)
            episodes.append(observations, actions, rewards, batch_ids)
            observations, batch_ids = new_observations, new_batch_ids
        return episodes

    def reset_task(self, task):
        tasks = [task for _ in range(self.num_workers)]
        reset = self.envs.reset_task(tasks)
        return all(reset)

    def sample_tasks(self, num_tasks, train=True):
        fn_signature = signature(self._env.unwrapped.sample_tasks)
        if 'train' in fn_signature.parameters: # to cater for meta-world environemnts
            tasks = self._env.unwrapped.sample_tasks(num_tasks, train)
        else: # to cater for other environments
            tasks = self._env.unwrapped.sample_tasks(num_tasks)
        return tasks

class BatchSamplerAnalysis(BatchSampler):
    def __init__(self, env_name, batch_size, device, seed, num_workers=mp.cpu_count() - 1,
                env_config_path=None):
        super(BatchSamplerAnalysis, self).__init__(env_name, batch_size, device, seed, num_workers,
            env_config_path)

    def sample(self, policy, params=None, gamma=0.95, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        episodes = BatchEpisodesAnalysis(batch_size=batch_size, gamma=gamma, device=self.device)
        for i in range(batch_size):
            self.queue.put(i)
        for _ in range(self.num_workers):
            self.queue.put(None)
        observations, batch_ids = self.envs.reset()
        dones = [False]
        while (not all(dones)) or (not self.queue.empty()):
            with torch.no_grad():
                observations_tensor = torch.from_numpy(observations).to(device=self.device)
                #actions_tensor = policy(observations_tensor, params=params).sample()
                actions_distribution, activations = policy.forward_analysismode(observations_tensor, params=params)
                actions_tensor = actions_distribution.sample()
                actions = actions_tensor.cpu().numpy()
                activations = activations.cpu().numpy()
            new_observations, rewards, dones, new_batch_ids, infos = self.envs.step(actions)
            if 'success' in infos[0].keys():
                success_metric = np.array([info['success'] for info in infos], dtype=np.float32)
                episodes.append(observations, activations, actions, rewards,batch_ids,success_metric)
            else:
                episodes.append(observations, activations, actions, rewards, batch_ids)
            observations, batch_ids = new_observations, new_batch_ids
        return episodes
