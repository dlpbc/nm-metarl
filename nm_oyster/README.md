## Neuromodulated PEARL

Code for "[Context Meta-Reinforcement Learning via Neuromodulation](https://arxiv.org/abs/2111.00134)" - 

- Prerequisites:
    - See the [original PEARL repo](https://github.com/katerakelly/oyster) for the pre-requisite.
    - For the ML1 and ML45 experiments you need [`meta-world`](https://github.com/rlworkgroup/metaworld)

### Running Experiments (procedures from the original PEARL repo)

Experiments are configured via `json` configuration files located in `./configs`. To reproduce an experiment, run:
`python launch_experiment.py ./configs/[EXP].json`

By default the code will use the GPU - to use CPU instead, set `use_gpu=False` in the appropriate config file.

Output files will be written to `./output/[ENV]/[EXP NAME]` where the experiment name is uniquely generated based on the date.
The file `progress.csv` contains statistics logged over the course of training.
We recommend `viskit` for visualizing learning curves: https://github.com/vitchyr/viskit

Network weights are also snapshotted during training.
To evaluate a learned policy after training has concluded, run `sim_policy.py`.
This script will run a given policy across a set of evaluation tasks and optionally generate a video of these trajectories.
Rendering is offline and the video is saved to the experiment folder.

#### Acknowledgements

Special thanks to Kate Rakelly
for her open-sourced PEARL implementation.
This was of great help to us, and our 
implementation are based on the PyTorch code from `https://github.com/katerakelly/oyster`
