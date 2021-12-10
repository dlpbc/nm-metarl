# CAVIA
# seed 42
# best (itr 1010)
CUDA_VISIBLE_DEVICES=4 python test.py \
./logs/CTgraphEnvGoalChange-v0/cavia/20_seed\=42fo\=Falselr\=0.5tau\=1.0_02_02_2021_19_43_22_cavia/config.json \
./saves/CTgraphEnvGoalChange-v0/cavia/20_seed\=42fo\=Falselr\=0.5tau\=1.0_02_02_2021_19_43_22_cavia/policy-1010.pt \
--env-config-path ./logs/CTgraphEnvGoalChange-v0/cavia/20_seed\=42fo\=Falselr\=0.5tau\=1.0_02_02_2021_19_43_22_cavia/ctgraph_d4.json

# seed 102
# best (itr 730)
CUDA_VISIBLE_DEVICES=4 python test.py \
./logs/CTgraphEnvGoalChange-v0/cavia/20_seed\=102fo\=Falselr\=0.5tau\=1.0_15_02_2021_13_32_37_cavia/config.json \
./saves/CTgraphEnvGoalChange-v0/cavia/20_seed\=102fo\=Falselr\=0.5tau\=1.0_15_02_2021_13_32_37_cavia/policy-730.pt \
--env-config-path ./logs/CTgraphEnvGoalChange-v0/cavia/20_seed\=102fo\=Falselr\=0.5tau\=1.0_15_02_2021_13_32_37_cavia/ctgraph_d4.json

# seed 378
# best (itr 780)
CUDA_VISIBLE_DEVICES=4 python test.py \
./logs/CTgraphEnvGoalChange-v0/cavia/20_seed\=378fo\=Falselr\=0.5tau\=1.0_23_02_2021_01_19_45_cavia/config.json \
./saves/CTgraphEnvGoalChange-v0/cavia/20_seed\=378fo\=Falselr\=0.5tau\=1.0_23_02_2021_01_19_45_cavia/policy-780.pt \
--env-config-path ./logs/CTgraphEnvGoalChange-v0/cavia/20_seed\=378fo\=Falselr\=0.5tau\=1.0_23_02_2021_01_19_45_cavia/ctgraph_d4.json

# NMCAVIA
# seed 42
# best (itr 1330)
CUDA_VISIBLE_DEVICES=4 python test.py \
./logs/CTgraphEnvGoalChange-v0/nm_cavia/20_seed\=42fo\=Falselr\=0.5tau\=1.0_02_02_2021_01_40_40_cavia/config.json \
./saves/CTgraphEnvGoalChange-v0/nm_cavia/20_seed\=42fo\=Falselr\=0.5tau\=1.0_02_02_2021_01_40_40_cavia/policy-1330.pt \
--env-config-path ./logs/CTgraphEnvGoalChange-v0/nm_cavia/20_seed\=42fo\=Falselr\=0.5tau\=1.0_02_02_2021_01_40_40_cavia/ctgraph_d4.json

# seed 102
# best (itr 720)
CUDA_VISIBLE_DEVICES=4 python test.py \
./logs/CTgraphEnvGoalChange-v0/nm_cavia/20_seed\=102fo\=Falselr\=0.5tau\=1.0_24_02_2021_17_42_22_cavia/config.json \
./saves/CTgraphEnvGoalChange-v0/nm_cavia/20_seed\=102fo\=Falselr\=0.5tau\=1.0_24_02_2021_17_42_22_cavia/policy-720.pt \
--env-config-path ./logs/CTgraphEnvGoalChange-v0/nm_cavia/20_seed\=102fo\=Falselr\=0.5tau\=1.0_24_02_2021_17_42_22_cavia/ctgraph_d4.json

# seed 378
# best (itr 1230)
CUDA_VISIBLE_DEVICES=4 python test.py \
./logs/CTgraphEnvGoalChange-v0/nm_cavia/20_seed\=378fo\=Falselr\=0.5tau\=1.0_01_03_2021_17_18_25_cavia/config.json \
./saves/CTgraphEnvGoalChange-v0/nm_cavia/20_seed\=378fo\=Falselr\=0.5tau\=1.0_01_03_2021_17_18_25_cavia/policy-1230.pt \
--env-config-path ./logs/CTgraphEnvGoalChange-v0/nm_cavia/20_seed\=378fo\=Falselr\=0.5tau\=1.0_01_03_2021_17_18_25_cavia/ctgraph_d4.json
