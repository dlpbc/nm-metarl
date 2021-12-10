# CAVIA
# seed 102
# best (itr 770)
python test.py \
./logs/CTgraphEnvGoalChange-v0/cavia/10_seed\=102fo\=Falselr\=0.5tau\=1.0_21_01_2021_04_08_22_cavia/config.json \
./saves/CTgraphEnvGoalChange-v0/cavia/10_seed\=102fo\=Falselr\=0.5tau\=1.0_21_01_2021_04_08_22_cavia/policy-770.pt \
--env-config-path ./logs/CTgraphEnvGoalChange-v0/cavia/10_seed\=102fo\=Falselr\=0.5tau\=1.0_21_01_2021_04_08_22_cavia/ctgraph_d3.json
# last saved model (itr 1000)
python test.py \
./logs/CTgraphEnvGoalChange-v0/cavia/10_seed\=102fo\=Falselr\=0.5tau\=1.0_21_01_2021_04_08_22_cavia/config.json \
./saves/CTgraphEnvGoalChange-v0/cavia/10_seed\=102fo\=Falselr\=0.5tau\=1.0_21_01_2021_04_08_22_cavia/policy-1000.pt \
--env-config-path ./logs/CTgraphEnvGoalChange-v0/cavia/10_seed\=102fo\=Falselr\=0.5tau\=1.0_21_01_2021_04_08_22_cavia/ctgraph_d3.json

# seed 42
# best (itr 940)
python test.py \
./logs/CTgraphEnvGoalChange-v0/cavia/10_seed\=42fo\=Falselr\=0.5tau\=1.0_21_01_2021_14_09_09_cavia/config.json \
./saves/CTgraphEnvGoalChange-v0/cavia/10_seed\=42fo\=Falselr\=0.5tau\=1.0_21_01_2021_14_09_09_cavia/policy-940.pt \
--env-config-path ./logs/CTgraphEnvGoalChange-v0/cavia/10_seed\=42fo\=Falselr\=0.5tau\=1.0_21_01_2021_14_09_09_cavia/ctgraph_d3.json
# last saved model (itr 1000)
python test.py \
./logs/CTgraphEnvGoalChange-v0/cavia/10_seed\=42fo\=Falselr\=0.5tau\=1.0_21_01_2021_14_09_09_cavia/config.json \
./saves/CTgraphEnvGoalChange-v0/cavia/10_seed\=42fo\=Falselr\=0.5tau\=1.0_21_01_2021_14_09_09_cavia/policy-1000.pt \
--env-config-path ./logs/CTgraphEnvGoalChange-v0/cavia/10_seed\=42fo\=Falselr\=0.5tau\=1.0_21_01_2021_14_09_09_cavia/ctgraph_d3.json

# seed 378
# best (itr 710)
python test.py \
./logs/CTgraphEnvGoalChange-v0/cavia/10_seed\=378fo\=Falselr\=0.5tau\=1.0_21_01_2021_17_22_00_cavia/config.json \
./saves/CTgraphEnvGoalChange-v0/cavia/10_seed\=378fo\=Falselr\=0.5tau\=1.0_21_01_2021_17_22_00_cavia/policy-710.pt \
--env-config-path ./logs/CTgraphEnvGoalChange-v0/cavia/10_seed\=378fo\=Falselr\=0.5tau\=1.0_21_01_2021_17_22_00_cavia/ctgraph_d3.json
# last saved model (itr 1000)
python test.py \
./logs/CTgraphEnvGoalChange-v0/cavia/10_seed\=378fo\=Falselr\=0.5tau\=1.0_21_01_2021_17_22_00_cavia/config.json \
./saves/CTgraphEnvGoalChange-v0/cavia/10_seed\=378fo\=Falselr\=0.5tau\=1.0_21_01_2021_17_22_00_cavia/policy-1000.pt \
--env-config-path ./logs/CTgraphEnvGoalChange-v0/cavia/10_seed\=378fo\=Falselr\=0.5tau\=1.0_21_01_2021_17_22_00_cavia/ctgraph_d3.json


# NMCAVIA
# seed 102
# best (iter 770)
CUDA_VISIBLE_DEVICES=4 python test.py \
./logs/CTgraphEnvGoalChange-v0/nm_cavia/10_seed\=102fo\=Falselr\=0.5tau\=1.0_20_01_2021_03_42_24_cavia/config.json \
./saves/CTgraphEnvGoalChange-v0/nm_cavia/10_seed\=102fo\=Falselr\=0.5tau\=1.0_20_01_2021_03_42_24_cavia/policy-770.pt \
--env-config-path ./logs/CTgraphEnvGoalChange-v0/nm_cavia/10_seed\=102fo\=Falselr\=0.5tau\=1.0_20_01_2021_03_42_24_cavia/ctgraph_d3.json
# final (iter 1000)
CUDA_VISIBLE_DEVICES=4 python test.py \
./logs/CTgraphEnvGoalChange-v0/nm_cavia/10_seed\=102fo\=Falselr\=0.5tau\=1.0_20_01_2021_03_42_24_cavia/config.json \
./saves/CTgraphEnvGoalChange-v0/nm_cavia/10_seed\=102fo\=Falselr\=0.5tau\=1.0_20_01_2021_03_42_24_cavia/policy-1000.pt \
--env-config-path ./logs/CTgraphEnvGoalChange-v0/nm_cavia/10_seed\=102fo\=Falselr\=0.5tau\=1.0_20_01_2021_03_42_24_cavia/ctgraph_d3.json

# seed 42
# best (iter 900)
CUDA_VISIBLE_DEVICES=4 python test.py \
./logs/CTgraphEnvGoalChange-v0/nm_cavia/10_seed\=42fo\=Falselr\=0.5tau\=1.0_20_01_2021_14_02_10_cavia/config.json \
./saves/CTgraphEnvGoalChange-v0/nm_cavia/10_seed\=42fo\=Falselr\=0.5tau\=1.0_20_01_2021_14_02_10_cavia/policy-900.pt \
--env-config-path ./logs/CTgraphEnvGoalChange-v0/nm_cavia/10_seed\=42fo\=Falselr\=0.5tau\=1.0_20_01_2021_14_02_10_cavia/ctgraph_d3.json
# final (iter 1000)
CUDA_VISIBLE_DEVICES=4 python test.py \
./logs/CTgraphEnvGoalChange-v0/nm_cavia/10_seed\=42fo\=Falselr\=0.5tau\=1.0_20_01_2021_14_02_10_cavia/config.json \
./saves/CTgraphEnvGoalChange-v0/nm_cavia/10_seed\=42fo\=Falselr\=0.5tau\=1.0_20_01_2021_14_02_10_cavia/policy-1000.pt \
--env-config-path ./logs/CTgraphEnvGoalChange-v0/nm_cavia/10_seed\=42fo\=Falselr\=0.5tau\=1.0_20_01_2021_14_02_10_cavia/ctgraph_d3.json

# seed 378
# best (iter 860)
CUDA_VISIBLE_DEVICES=4 python test.py \
./logs/CTgraphEnvGoalChange-v0/nm_cavia/10_seed\=378fo\=Falselr\=0.5tau\=1.0_20_01_2021_23_39_33_cavia/config.json \
./saves/CTgraphEnvGoalChange-v0/nm_cavia/10_seed\=378fo\=Falselr\=0.5tau\=1.0_20_01_2021_23_39_33_cavia/policy-860.pt \
--env-config-path ./logs/CTgraphEnvGoalChange-v0/nm_cavia/10_seed\=378fo\=Falselr\=0.5tau\=1.0_20_01_2021_23_39_33_cavia/ctgraph_d3.json
# final (iter 1000)
CUDA_VISIBLE_DEVICES=4 python test.py \
./logs/CTgraphEnvGoalChange-v0/nm_cavia/10_seed\=378fo\=Falselr\=0.5tau\=1.0_20_01_2021_23_39_33_cavia/config.json \
./saves/CTgraphEnvGoalChange-v0/nm_cavia/10_seed\=378fo\=Falselr\=0.5tau\=1.0_20_01_2021_23_39_33_cavia/policy-1000.pt \
--env-config-path ./logs/CTgraphEnvGoalChange-v0/nm_cavia/10_seed\=378fo\=Falselr\=0.5tau\=1.0_20_01_2021_23_39_33_cavia/ctgraph_d3.json
