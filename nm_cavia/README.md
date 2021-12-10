## Neuromodulated CAVIA

Code for "[Context Meta-Reinforcement Learning via Neuromodulation](https://arxiv.org/abs/2111.00134)"

Python 3.6 and PyTorch 1.2.0 were used for these experiments.

### Meta-Reinforcement Learning

This code is an extension of the original [CAVIA Meta-RL implementation](https://github.com/lmzintgraf/cavia/tree/master/rl).

- Prerequisites (see requirements.txt):
    - numpy
    - pytorch
    - gym
    - tensorboardX
    - scikit-image
    - For the CT-graph experiments you need [`ct-graph`](https://github.com/soltoggio/CT-graph)
    - For the Half-Cheetah MuJoCo experiments you need [`mujoco-py`](https://github.com/openai/mujoco-py)
    - For the ML1 and ML45 experiments you need [`meta-world`](https://github.com/rlworkgroup/metaworld)

    Note: consider setting up a conda or virtualenv environments to install packages

- Running experiments:

    To run an experiments using the neuromodulated policy, use the following commands:

	*2D navigation*
    ```
    $ cd /path/to/codebase/rl 
	$ python main.py --env-name 2DNavigation-v0 \ 
	--num-context-params 5 --hidden-size 100 --nonlinearity relu \
    --fast-batch-size 20 --fast-lr 0.5 --meta-batch-size 20 --test-batch-size 40 --num-test-steps 4 \
    --num-workers 4 --num-batches 501 --test-freq 10 --seed 42 \
    --expname cavia --neuromodulation --nm-size 4
    ```

	*Half-Cheetah direction*
	```
    $ cd /path/to/codebase/rl 
	$ python main.py --env-name HalfCheetahDir-v1 \
	--num-context-params 50 --hidden-size 200 --num-layers 2 --nonlinearity relu \
	--fast-batch-size 20 --fast-lr 10.0 --meta-batch-size 40 --test-batch-size 40 --num-test-steps 4 \
	--num-workers 4 --num-batches 501 --test-freq 10 --seed 378 \
	--expname nmcavia-softgating --neuromodulation --nm-size 32 --nm-gate soft
	```

	*Half-Cheetah velocity*
	```
    $ cd /path/to/codebase/rl 
    $ python main.py --env-name HalfCheetahVel-v1 \
	--num-context-params 50 --hidden-size 200 --num-layers 2 --nonlinearity relu \
	--fast-batch-size 20 --fast-lr 10.0 --meta-batch-size 40 --test-batch-size 40 --num-test-steps 4 \
	--num-workers 4 --num-batches 501 --test-freq 10 --seed 42 \
	--expname nmcavia-softgating --neuromodulation --nm-size 32 --nm-gate soft
	```

	*Meta-world ML1 (push)*
    ```
    $ cd /path/to/codebase/rl 
	$ python main.py --env-name ML1Push-v2 \
    --num-context-params 50 --hidden-size 200 --num-layers 2 --nonlinearity relu \
    --fast-batch-size 20 --fast-lr 10.0 --meta-batch-size 40 --test-batch-size 40 --num-test-steps 4 \
	--num-workers 4 --num-batches 501 --test-freq 10 --seed 42 \
	--expname nmcavia-softgating --neuromodulation --nm-size 32 --nm-gate soft
    ```

	*Meta-world ML45*
	```
    $ cd /path/to/codebase/rl 
	$ python main.py --env-name ML45-v2 \
	--num-context-params 100 --hidden-size 200 --num-layers 2 --nonlinearity relu \
	--fast-batch-size 10 --fast-lr 10.0 --meta-batch-size 45 --test-batch-size 20 --num-test-steps 4 \
	--num-workers 4 --num-batches 501 --test-freq 10 --seed 378 \
	--expname nmcavia-softgating --neuromodulation --nm-size 32 --nm-gate soft
	```

	*CT-graph depth2*
    ```
    $ cd /path/to/codebase/rl
    $ python main.py --env-name CTgraphEnvGoalChange-v0 \
    --num-context-params 5 --hidden-size 200 --nonlinearity relu \
    --fast-batch-size 20 --fast-lr 0.5 --meta-batch-size 20 --test-batch-size 40 --num-test-steps 4 \
    --num-workers 4 --num-batches 501 --test-freq 10 --seed 42 \
    --expname cavia --env-config-path ./envs/ctgraph/ctgraph_d2.json --neuromodulation --nm-size 8
    ```

    Note: 
        - to run an experiment with the standard policy network (default CAVIA), remove the
          `--neuromodulation` and `--nm-size` flags from the above commands.
        - also, to run other CTgraph experiments (e.g. depth 3 and 4), change the value of the
          `--env-config-path` to `./envs/ctgraph/ctgraph_d3.json` for depth 3 and 
          `./envs/ctgraph/ctgraph_d4.json` for depth 4.

    Lastly, to evaluate adaptation capability of a trained policy, use the following command:

    ```
    python test.py \
    logs/env_key/path/to/experiment/config.json \
    saves/env_key/path/to/experiment/policy-<iteration_number>.pt
    ```
    for the 2D navigation, half-cheetah and meta-world environments

    or

    ```
    python test.py \
    logs/env_key/path/to/experiment/config.json \
    saves/env_key/path/to/experiment/policy-<iteration_number>.pt
    --env-config-path logs/env_key/path/to/experiment/ctgraph_d<number>.json
    ```
    for the CTgraph environment.

#### Acknowledgements

Special thanks to Luisa M Zintgraf
for her open-sourced CAVIA implementation.
This was of great help to us, and parts of our 
implementation are based on the PyTorch code from `https://github.com/lmzintgraf/cavia`

