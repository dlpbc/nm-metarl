import argparse
import shutil
import datetime
import json
import os
#import matplotlib.pyplot as plt
import time
from types import SimpleNamespace

import numpy as np
import scipy.stats as st
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter

import utils
from arguments import parse_args
from baseline import LinearFeatureBaseline
from metalearner import MetaLearner
from policies.categorical_mlp import CategoricalMLPPolicy, CaviaCategoricalMLPPolicy
from policies.categorical_mlp import NMCategoricalMLPPolicy, NMCaviaCategoricalMLPPolicy
from policies.normal_mlp import NormalMLPPolicy, CaviaMLPPolicy, NMNormalMLPPolicy, NMCaviaMLPPolicy
from sampler import BatchSampler, BatchSamplerAnalysis

def get_success_metric(episodes_per_task):

    success_metric = []
    for task_idx in range(len(episodes_per_task)):
        curr_success_metric = []
        episodes = episodes_per_task[task_idx]
        for update_idx in range(len(episodes)):
            # max_episode_length x num_evals/batch_size_per_task
            ret = episodes[update_idx].success_metric
            curr_success_metric.append(ret)
        # result will be: max_episode_length x num_evals x num_updates
        #success_metric.append(torch.stack(curr_success_metric, dim=2))
        success_metric.append(np.stack(curr_success_metric, axis=2))

    # result will be: num_tasks x max_episode_length x num_evals x num_updates
    #success_metric = torch.stack(success_metric)
    success_metric = np.stack(success_metric)
    return success_metric

def get_raw_activations(episodes_per_task):
    activations = []
    for task_idx in range(len(episodes_per_task)):
        curr_activations = []
        episodes = episodes_per_task[task_idx]
        for update_idx in range(len(episodes)):
            # max_episode_length x num_evals/batch_size_per_task x activations_dim
            # activations_dim is sum of all hidden features dim and output dim
            #ret = episodes[update_idx].activations * episodes[update_idx].mask

            ret = episodes[update_idx].activations
            #ret = episodes[update_idx].activations.detach().cpu()
            curr_activations.append(ret)
        # result will be: max_episode_length x num_evals x activations_dim x num_updates
        #activations.append(torch.stack(curr_activations, dim=3))
        activations.append(np.stack(curr_activations, axis=3))

    # result will be: num_tasks x max_episode_length x num_evals x activations_dim x num_updates
    #activations = torch.stack(activations)
    activations = np.stack(activations)
    return activations

def get_raw_reward(episodes_per_task):

    rewards = []
    for task_idx in range(len(episodes_per_task)):
        curr_rewards = []
        episodes = episodes_per_task[task_idx]
        for update_idx in range(len(episodes)):
            # max_episode_length x num_evals/batch_size_per_task
            ret = episodes[update_idx].rewards * episodes[update_idx].mask
            curr_rewards.append(ret)
        # result will be: max_episode_length x num_evals x num_updates
        rewards.append(torch.stack(curr_rewards, dim=2))

    # result will be: num_tasks x max_episode_length x num_evals x num_updates
    rewards = torch.stack(rewards)
    return rewards.cpu().numpy()

def get_returns(episodes_per_task):

    # sum up for each rollout, then take the mean across rollouts
    returns = []
    for task_idx in range(len(episodes_per_task)):
        curr_returns = []
        episodes = episodes_per_task[task_idx]
        for update_idx in range(len(episodes)):
            # compute returns for individual rollouts
            ret = (episodes[update_idx].rewards * episodes[update_idx].mask).sum(dim=0)
            curr_returns.append(ret)
        # result will be: num_evals/batch_size_per_task * num_updates
        returns.append(torch.stack(curr_returns, dim=1))

    # result will be: num_tasks * num_evals * num_updates
    returns = torch.stack(returns)
    returns = returns.reshape((-1, returns.shape[-1]))

    return returns

def total_rewards(episodes_per_task, interval=False):

    returns = get_returns(episodes_per_task).cpu().numpy()

    mean = np.mean(returns, axis=0)
    conf_int = st.t.interval(0.95, len(mean) - 1, loc=mean, scale=st.sem(returns, axis=0))
    conf_int = mean - conf_int
    if interval:
        return mean, conf_int[0]
    else:
        return mean


def main(args):
    print('starting....')

    utils.set_seed(args.seed, cudnn=args.make_deterministic)

    continuous_actions = (args.env_name in ['AntVel-v1', 'AntDir-v1',
                                            'AntPos-v0', 'HalfCheetahVel-v1', 'HalfCheetahDir-v1',
                                            '2DNavigation-v0',
                                            'ML1Push-v2', 'ML45-v2'])

    prefix = 'nm_' if args.neuromodulation else ''
    # subfolders for logging
    method_used = 'maml_test' if args.maml else 'cavia_test'
    method_used = prefix + method_used
    num_context_params = str(args.num_context_params) + '_' if not args.maml else ''
    output_name = num_context_params + 'seed=' + str(args.seed) + 'fo=' + str(args.first_order)
    output_name += 'lr=' + str(args.fast_lr) + 'tau=' + str(args.tau)
    output_name += '_' + datetime.datetime.now().strftime('%d_%m_%Y_%H_%M_%S')
    output_name += '_' + args.expname
    dir_path = os.path.dirname(os.path.realpath(__file__))
    log_folder = os.path.join(os.path.join(dir_path, 'logs'), args.env_name, method_used, output_name)
    save_folder = os.path.join(os.path.join(dir_path, 'saves'), args.env_name, method_used, output_name)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    if args.env_config_path is not None:
        shutil.copy(args.env_config_path, log_folder)

    # initialise tensorboard writer
    writer = SummaryWriter(log_folder)

    # save config file
    with open(os.path.join(save_folder, 'config.json'), 'w') as f:
        config = {k: v for (k, v) in vars(args).items() if k != 'device'}
        config.update(device=args.device.type)
        json.dump(config, f, indent=2)
    with open(os.path.join(log_folder, 'config.json'), 'w') as f:
        config = {k: v for (k, v) in vars(args).items() if k != 'device'}
        config.update(device=args.device.type)
        json.dump(config, f, indent=2)

    sampler = BatchSamplerAnalysis(args.env_name, batch_size=args.fast_batch_size, num_workers=args.num_workers,
                           device=args.device, seed=args.seed, env_config_path=args.env_config_path)

    if args.nonlinearity == 'relu':
        nonlinearity = F.relu
    elif args.nonlinearity == 'tanh':
        nonlinearity = torch.tanh
    else:
        raise ValueError('invalid activation in `args.activation`')

    if continuous_actions:
        if not args.maml:
            if not args.neuromodulation:
                print('continuous actions / cavia / without neuromodulation')
                policy = CaviaMLPPolicy(
                    int(np.prod(sampler.envs.observation_space.shape)),
                    int(np.prod(sampler.envs.action_space.shape)),
                    hidden_sizes=(args.hidden_size,) * args.num_layers,
                    num_context_params=args.num_context_params,
                    device=args.device,
                    nonlinearity=nonlinearity
                )
            else:
                print('continuous actions / cavia / with neuromodulation')
                policy = NMCaviaMLPPolicy(
                    int(np.prod(sampler.envs.observation_space.shape)),
                    int(np.prod(sampler.envs.action_space.shape)),
                    hidden_sizes=(args.hidden_size,) * args.num_layers,
                    num_context_params=args.num_context_params,
                    device=args.device,
                    nm_size=args.nm_size,
                    nm_gate=args.nm_gate,
                    nonlinearity=nonlinearity
                )
        else:
            print('first order: ', args.first_order)
            if not args.neuromodulation:
                print('continuous actions / maml / without neuromodulation')
                policy = NormalMLPPolicy(
                    int(np.prod(sampler.envs.observation_space.shape)),
                    int(np.prod(sampler.envs.action_space.shape)),
                    hidden_sizes=(args.hidden_size,) * args.num_layers,
                    nonlinearity=nonlinearity
                )
            else:
                print('continuous actions / maml / with neuromodulation')
                policy = NMNormalMLPPolicy(
                    int(np.prod(sampler.envs.observation_space.shape)),
                    int(np.prod(sampler.envs.action_space.shape)),
                    hidden_sizes=(args.hidden_size,) * args.num_layers,
                    nm_size=args.nm_size,
                    nm_gate=args.nm_gate,
                    nonlinearity=nonlinearity
                )
    else:
        if not args.maml:
            if not args.neuromodulation:  
                print('discrete actions / cavia / without neuromodulation')
                policy = CaviaCategoricalMLPPolicy(
                    int(np.prod(sampler.envs.observation_space.shape)),
                    sampler.envs.action_space.n,
                    hidden_sizes=(args.hidden_size,) * args.num_layers,
                    num_context_params=args.num_context_params,
                    device=args.device,
                    nonlinearity=nonlinearity
                )
            else:
                print('discrete actions / cavia / with neuromodulation')
                policy = NMCaviaCategoricalMLPPolicy(
                    int(np.prod(sampler.envs.observation_space.shape)),
                    sampler.envs.action_space.n,
                    hidden_sizes=(args.hidden_size,) * args.num_layers,
                    num_context_params=args.num_context_params,
                    device=args.device,
                    nm_size=args.nm_size,
                    nm_gate=args.nm_gate,
                    nonlinearity=nonlinearity
                )
        else:
            if not args.neuromodulation:
                print('discrete actions / maml / without neuromodulation')
                policy = CategoricalMLPPolicy(
                    int(np.prod(sampler.envs.observation_space.shape)),
                    sampler.envs.action_space.n,
                    hidden_sizes=(args.hidden_size,) * args.num_layers,
                    nonlinearity=nonlinearity
                )
            else:
                print('discrete actions / maml / with neuromodulation')
                policy = NMCategoricalMLPPolicy(
                    int(np.prod(sampler.envs.observation_space.shape)),
                    sampler.envs.action_space.n,
                    hidden_sizes=(args.hidden_size,) * args.num_layers,
                    nm_size=args.nm_size,
                    nm_gate=args.nm_gate,
                    nonlinearity=nonlinearity
                )

    # load trained policy param
    policy.load_state_dict(torch.load(args.model_path))
    #policy.load_state_dict(torch.load(args.model_path, map_location=torch.device('cpu')))


    # initialise baseline
    baseline = LinearFeatureBaseline(int(np.prod(sampler.envs.observation_space.shape)))

    # initialise meta-learner
    metalearner = MetaLearner(sampler, policy, baseline, gamma=args.gamma, fast_lr=args.fast_lr, tau=args.tau,
                              device=args.device)

    # -- evaluation. evaluate for multiple update steps
    test_tasks = sampler.sample_tasks(num_tasks=args.meta_batch_size)
    #test_tasks = sampler._env.all_tasks # NOTE
    test_episodes = metalearner.test(test_tasks, num_steps=args.num_test_steps,
                                     batch_size=args.test_batch_size, halve_lr=args.halve_test_lr)
    all_returns = total_rewards(test_episodes, interval=True)
    for num in range(args.num_test_steps + 1):
        #writer.add_scalar('evaluation_rew/avg_rew ' + str(num), all_returns[0][num], batch)
        #writer.add_scalar('evaluation_cfi/avg_rew ' + str(num), all_returns[1][num], batch)
        writer.add_scalar('evaluation_rew/avg_rew ', all_returns[0][num], num)
        writer.add_scalar('evaluation_cfi/avg_rew ', all_returns[1][num], num)
    # get and save raw rewards
    # shape: num_tasks x max_episode_length x batch_size x num_updates
    # note: batch_size = args.test_batch_size
    # note: num_updates = args.num_test_steps + 1 (1 because of initial sample before any update)
    raw_rewards = get_raw_reward(test_episodes)
    print('raw_rewards.shape:', raw_rewards.shape)
    print('average reward:', raw_rewards.sum(axis=1).mean())
    np.save(os.path.join(save_folder, 'rewards.npy'), raw_rewards)
    # activations
    # shape: num_tasks x max_episode_length x batch_size x activations_dim x num_updates
    # note; activations_dim is sum of all hidden features dim and output dim
    activations = get_raw_activations(test_episodes)
    print('activations.shape:', activations.shape)
    np.save(os.path.join(save_folder, 'activations.npy'), activations)
    # save tasks details
    np.save(os.path.join(save_folder, 'tasks.npy'), np.array(test_tasks))
    # save success metric if environment is a metaworld env instance
    if 'ML' in args.env_name:
        success_metric = get_success_metric(test_episodes)
        np.save(os.path.join(save_folder, 'success_metric.npy'), np.array(success_metric))


    # -- save policy network
    with open(os.path.join(save_folder, 'policy.pt'), 'wb') as f:
        torch.save(policy.state_dict(), f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fast Context Adaptation via Meta-Learning (CAVIA)')
    parser.add_argument('config_path', type=str,
                        help='path/to/experiment/configuration')
    parser.add_argument('model_path', type=str,
                        help='path/to/saved/model')
    parser.add_argument('--env-config-path', type=str,
                        default=None,
                        help='path/to/experiment environment configuration')
    args = parser.parse_args()

    with open(args.config_path) as f:
        config = json.load(f)
    if args.env_config_path is not None:
        config['env_config_path'] = args.env_config_path
    config['model_path'] = args.model_path
    config['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # override number of workers in config
    config['num_workers'] = 1
    #config['meta_batch_size'] = 8
    if 'nonlinearity' not in config.keys():
        print('nonlinearity not found in config. defaulting to relu')
        config['nonlinearity'] = F.relu
    args = SimpleNamespace(**config)

    main(args)
