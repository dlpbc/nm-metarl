from gym.envs.registration import register

# Bandit
# ----------------------------------------

for k in [5, 10, 50]:
    register(
        'Bandit-K{0}-v0'.format(k),
        entry_point='envs.bandit:BernoulliBanditEnv',
        kwargs={'k': k}
    )

# TabularMDP
# ----------------------------------------

register(
    'TabularMDP-v0',
    entry_point='envs.mdp:TabularMDPEnv',
    kwargs={'num_states': 10, 'num_actions': 5},
    max_episode_steps=10
)

# Mujoco
# ----------------------------------------

register(
    'AntVel-v1',
    entry_point='envs.utils:mujoco_wrapper',
    kwargs={'entry_point': 'envs.mujoco.ant:AntVelEnv'},
    max_episode_steps=200
)

register(
    'AntDir-v1',
    entry_point='envs.utils:mujoco_wrapper',
    kwargs={'entry_point': 'envs.mujoco.ant:AntDirEnv'},
    max_episode_steps=200
)

register(
    'AntPos-v0',
    entry_point='envs.utils:mujoco_wrapper',
    kwargs={'entry_point': 'envs.mujoco.ant:AntPosEnv'},
    max_episode_steps=200
)

register(
    'HalfCheetahVel-v1',
    entry_point='envs.utils:mujoco_wrapper',
    kwargs={'entry_point': 'envs.mujoco.half_cheetah:HalfCheetahVelEnv'},
    max_episode_steps=200
)

register(
    'HalfCheetahDir-v1',
    entry_point='envs.utils:mujoco_wrapper',
    kwargs={'entry_point': 'envs.mujoco.half_cheetah:HalfCheetahDirEnv'},
    max_episode_steps=200
)

# 2D Navigation
# ----------------------------------------

register(
    '2DNavigation-v0',
    entry_point='envs.navigation:Navigation2DEnv',
    max_episode_steps=100
)

# CT-graph
# ----------------------------------------
register(
    id='CTgraphEnvGoalChange-v0',
    entry_point='envs.ctgraph:CTgraphEnvGoalChange',
)

# Minecraft
# ----------------------------------------
register(
    id='MCEnvGoalChange-v0',
    entry_point='envs.minecraft:MCEnvGoalChange',
)

# Meta-World
# ----------------------------------------
register(
    id='ML1Push-v2',
    entry_point='envs.utils:mujoco_wrapper',
    kwargs={'entry_point': 'envs.mujoco.mw_ml1:ML1Env', 'benchmark_name': 'push-v2'},
    max_episode_steps=200
)
register(
    id='ML45-v2',
    entry_point='envs.utils:mujoco_wrapper',
    kwargs={'entry_point': 'envs.mujoco.mw_ml45:ML45Env'},
    max_episode_steps=200
)
