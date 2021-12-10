# CAVIA
# seed 42
# best (itr 1200)
CUDA_VISIBLE_DEVICES=3 python test.py \
./logs/CTgraphEnvGoalChange-v0/cavia/20_seed\=42fo\=Falselr\=0.5tau\=1.0_30_05_2021_15_50_34_cavia/config.json \
./saves/CTgraphEnvGoalChange-v0/cavia/20_seed\=42fo\=Falselr\=0.5tau\=1.0_30_05_2021_15_50_34_cavia/policy-1200.pt \
--env-config-path ./logs/CTgraphEnvGoalChange-v0/cavia/20_seed\=42fo\=Falselr\=0.5tau\=1.0_30_05_2021_15_50_34_cavia/ctgraph_d4.json

# seed 102
# best (itr 1150)
CUDA_VISIBLE_DEVICES=3 python test.py \
./logs/CTgraphEnvGoalChange-v0/cavia/20_seed\=102fo\=Falselr\=0.5tau\=1.0_31_05_2021_19_52_01_cavia/config.json \
./saves/CTgraphEnvGoalChange-v0/cavia/20_seed\=102fo\=Falselr\=0.5tau\=1.0_31_05_2021_19_52_01_cavia/policy-1150.pt \
--env-config-path ./logs/CTgraphEnvGoalChange-v0/cavia/20_seed\=102fo\=Falselr\=0.5tau\=1.0_31_05_2021_19_52_01_cavia/ctgraph_d4.json

# seed 378
# best (itr 1500)
CUDA_VISIBLE_DEVICES=3 python test.py \
./logs/CTgraphEnvGoalChange-v0/cavia/20_seed\=378fo\=Falselr\=0.5tau\=1.0_31_05_2021_19_53_17_cavia/config.json \
./saves/CTgraphEnvGoalChange-v0/cavia/20_seed\=378fo\=Falselr\=0.5tau\=1.0_31_05_2021_19_53_17_cavia/policy-1500.pt \
--env-config-path ./logs/CTgraphEnvGoalChange-v0/cavia/20_seed\=378fo\=Falselr\=0.5tau\=1.0_31_05_2021_19_53_17_cavia/ctgraph_d4.json

