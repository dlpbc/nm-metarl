import datetime
import dateutil.tz
import os, shutil
import os.path as osp
import pickle
import json
import numpy as np
import click
import torch

from rlkit.envs import ENVS
from rlkit.envs.wrappers import NormalizedBoxEnv, CameraWrapper
from rlkit.torch.sac.policies import TanhGaussianPolicy, NMTanhGaussianPolicy
from rlkit.torch.networks import FlattenMlp, MlpEncoder, RecurrentEncoder
from rlkit.torch.sac.agent import PEARLAgent
from configs.default import default_config
from launch_experiment import deep_update_dict
from rlkit.torch.sac.policies import MakeDeterministic
from rlkit.samplers.util import rollout
from rlkit.launchers.launcher_util import setup_logger
from rlkit.core import logger


def sim_policy(variant, path_to_exp, num_trajs=1, deterministic=False, save_video=False):
    '''
    simulate a trained policy adapting to a new task
    optionally save videos of the trajectories - requires ffmpeg

    :variant: experiment configuration dict
    :path_to_exp: path to exp folder
    :num_trajs: number of trajectories to simulate per task (default 1)
    :deterministic: if the policy is deterministic (default stochastic)
    :save_video: whether to generate and save a video (default False)
    '''

    # create multi-task environment and sample tasks
    if variant['env_name'] in ['point-robot-v0', 'point-robot', 'sparse-point-robot']:
        env = NormalizedBoxEnv(ENVS[variant['env_name']](**variant['env_params']))
    else:
        env = CameraWrapper(NormalizedBoxEnv(ENVS[variant['env_name']](**variant['env_params'])), variant['util_params']['gpu_id'])
    tasks = env.get_all_task_idx()
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    eval_tasks=list(tasks[-variant['n_eval_tasks']:])
    print('testing on {} test tasks, {} trajectories each'.format(len(eval_tasks), num_trajs))

    # instantiate networks
    latent_dim = variant['latent_size']
    context_encoder = latent_dim * 2 if variant['algo_params']['use_information_bottleneck'] else latent_dim
    reward_dim = 1
    net_size = variant['net_size']
    recurrent = variant['algo_params']['recurrent']
    encoder_model = RecurrentEncoder if recurrent else MlpEncoder

    context_encoder = encoder_model(
        hidden_sizes=[200, 200, 200],
        input_size=obs_dim + action_dim + reward_dim,
        output_size=context_encoder,
    )

    nm = variant['util_params'].get('nm', False)
    if nm:
        print('using a neuromodulated policy.')
        policy = NMTanhGaussianPolicy(
            hidden_sizes=[net_size, net_size, net_size],
            obs_dim=obs_dim + latent_dim,
            latent_dim=latent_dim,
            action_dim=action_dim,
            nm_dim=variant['util_params']['nmsize']
        )
    else:
        print('using a standard policy.')
        policy = TanhGaussianPolicy(
            hidden_sizes=[net_size, net_size, net_size],
            obs_dim=obs_dim + latent_dim,
            latent_dim=latent_dim,
            action_dim=action_dim,
        )
    agent = PEARLAgent(
        latent_dim,
        context_encoder,
        policy,
        **variant['algo_params']
    )
    # deterministic eval
    if deterministic:
        agent = MakeDeterministic(agent)

    # load trained weights (otherwise simulate random policy)
    context_encoder.load_state_dict(torch.load(os.path.join(path_to_exp, 'context_encoder.pth')))
    policy.load_state_dict(torch.load(os.path.join(path_to_exp, 'policy.pth')))

    # create logging directory
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    if variant['util_params']['exp_id'] is not None:
        exp_id = '{0}_{1}'.format(timestamp, variant['util_params']['exp_id'])
    else:
        exp_id = '{0}'.format(timestamp)
    test_log_dir = setup_logger(variant['env_name'], variant=variant, exp_id=exp_id, base_log_dir=variant['util_params']['base_log_dir']+'/tests', log_tabular_only=False)

    # loop through tasks collecting rollouts
    all_rets = []
    success_metric = []
    video_frames = []
    for idx in eval_tasks:
        env.reset_task(idx)
        agent.clear_z()
        paths = []
        for n in range(num_trajs):
            path = rollout(env, agent, max_path_length=variant['algo_params']['max_path_length'], accum_context=True, save_frames=save_video)
            paths.append(path)
            if save_video:
                video_frames += [t['frame'] for t in path['env_infos']]
            if n >= variant['algo_params']['num_exp_traj_eval']:
                agent.infer_posterior(agent.context)
        all_rets.append([sum(p['rewards']) for p in paths])
        if 'success' in paths[0]['env_infos'][0].keys():
            trajs_success = []
            for p in paths:
                success = 0.
                for step_info in p['env_infos']:
                    if step_info['success']:
                        success = 1.
                        break
                trajs_success.append(success)
            success_metric.append(trajs_success)

    if save_video:
        # save frames to file temporarily
        temp_dir = os.path.join(path_to_exp, 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        for i, frm in enumerate(video_frames):
            frm.save(os.path.join(temp_dir, '%06d.jpg' % i))

        video_filename=os.path.join(path_to_exp, 'video.mp4'.format(idx))
        # run ffmpeg to make the video
        os.system('ffmpeg -i {}/%06d.jpg -vcodec mpeg4 {}'.format(temp_dir, video_filename))
        # delete the frames
        shutil.rmtree(temp_dir)


    # compute average returns across tasks
    n = min([len(a) for a in all_rets])
    rets = [a[:n] for a in all_rets]
    rets = np.mean(np.stack(rets), axis=0)
    for i, ret in enumerate(rets):
        #print('trajectory {}, avg return: {} \n'.format(i, ret))
        logger.record_tabular('trajectory', i)
        logger.record_tabular('avg_reward', float(ret))
        logger.dump_tabular(with_prefix=False, with_timestamp=False)

    # save rewards
    to_save = np.stack(all_rets)
    print('to_save.shape:', to_save.shape)
    np.save('{0}/rewards.npy'.format(test_log_dir), to_save)

    # save task
    tasks_ = np.array([env.tasks[idx] for idx in eval_tasks])
    np.save('{0}/tasks.npy'.format(test_log_dir), tasks_)

    # save success metric if it exists (note, metaworld only)
    if len(success_metric) > 0:
        success_metric = np.stack(success_metric)
        print('success_metric.shape:', success_metric.shape)
        np.save('{0}/success_metric.npy'.format(test_log_dir), success_metric)


@click.command()
@click.argument('config', default=None)
@click.argument('path', default=None)
@click.option('--num_trajs', default=3)
@click.option('--deterministic', is_flag=True, default=False)
@click.option('--video', is_flag=True, default=False)
@click.option('--meta_batch', default=None)
def main(config, path, num_trajs, deterministic, video, meta_batch):
    variant = default_config
    if config:
        with open(osp.join(config)) as f:
            exp_params = json.load(f)
        variant = deep_update_dict(exp_params, variant)
    if meta_batch is not None:
        variant['algo_params']['meta_batch'] = int(meta_batch)
        variant['n_train_tasks'] = 0
        variant['n_eval_tasks'] = int(meta_batch)
        variant['env_params']['n_tasks'] = int(meta_batch)
    sim_policy(variant, path, num_trajs, deterministic, video)


if __name__ == "__main__":
    main()
