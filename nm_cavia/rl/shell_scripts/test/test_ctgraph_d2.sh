# CAVIA
# seed 102
# best (itr 250)
python test.py \
./logs/CTgraphEnvGoalChange-v0/cavia/5_seed\=102fo\=Falselr\=0.5tau\=1.0_22_01_2021_04_21_38_cavia/config.json \
./saves/CTgraphEnvGoalChange-v0/cavia/5_seed\=102fo\=Falselr\=0.5tau\=1.0_22_01_2021_04_21_38_cavia/policy-250.pt \
--env-config-path ./logs/CTgraphEnvGoalChange-v0/cavia/5_seed\=102fo\=Falselr\=0.5tau\=1.0_22_01_2021_04_21_38_cavia/ctgraph.json
# last saved model (itr 500)
python test.py \
./logs/CTgraphEnvGoalChange-v0/cavia/5_seed\=102fo\=Falselr\=0.5tau\=1.0_22_01_2021_04_21_38_cavia/config.json \
./saves/CTgraphEnvGoalChange-v0/cavia/5_seed\=102fo\=Falselr\=0.5tau\=1.0_22_01_2021_04_21_38_cavia/policy-500.pt \
--env-config-path ./logs/CTgraphEnvGoalChange-v0/cavia/5_seed\=102fo\=Falselr\=0.5tau\=1.0_22_01_2021_04_21_38_cavia/ctgraph.json

# seed 42
# best (itr 350)
python test.py \
./logs/CTgraphEnvGoalChange-v0/cavia/5_seed\=42fo\=Falselr\=0.5tau\=1.0_22_01_2021_04_55_47_cavia/config.json \
./saves/CTgraphEnvGoalChange-v0/cavia/5_seed\=42fo\=Falselr\=0.5tau\=1.0_22_01_2021_04_55_47_cavia/policy-350.pt \
--env-config-path ./logs/CTgraphEnvGoalChange-v0/cavia/5_seed\=42fo\=Falselr\=0.5tau\=1.0_22_01_2021_04_55_47_cavia/ctgraph.json
# last saved model (itr 500)
python test.py \
./logs/CTgraphEnvGoalChange-v0/cavia/5_seed\=42fo\=Falselr\=0.5tau\=1.0_22_01_2021_04_55_47_cavia/config.json \
./saves/CTgraphEnvGoalChange-v0/cavia/5_seed\=42fo\=Falselr\=0.5tau\=1.0_22_01_2021_04_55_47_cavia/policy-500.pt \
--env-config-path ./logs/CTgraphEnvGoalChange-v0/cavia/5_seed\=42fo\=Falselr\=0.5tau\=1.0_22_01_2021_04_55_47_cavia/ctgraph.json

# seed 378
# best (itr 50)
python test.py \
./logs/CTgraphEnvGoalChange-v0/cavia/5_seed\=378fo\=Falselr\=0.5tau\=1.0_22_01_2021_14_02_14_cavia/config.json \
./saves/CTgraphEnvGoalChange-v0/cavia/5_seed\=378fo\=Falselr\=0.5tau\=1.0_22_01_2021_14_02_14_cavia/policy-50.pt \
--env-config-path ./logs/CTgraphEnvGoalChange-v0/cavia/5_seed\=378fo\=Falselr\=0.5tau\=1.0_22_01_2021_14_02_14_cavia/ctgraph.json
# last saved model (itr 500)
python test.py \
./logs/CTgraphEnvGoalChange-v0/cavia/5_seed\=378fo\=Falselr\=0.5tau\=1.0_22_01_2021_14_02_14_cavia/config.json \
./saves/CTgraphEnvGoalChange-v0/cavia/5_seed\=378fo\=Falselr\=0.5tau\=1.0_22_01_2021_14_02_14_cavia/policy-500.pt \
--env-config-path ./logs/CTgraphEnvGoalChange-v0/cavia/5_seed\=378fo\=Falselr\=0.5tau\=1.0_22_01_2021_14_02_14_cavia/ctgraph.json


# NMCAVIA
# seed 102
# best (iter 250)
CUDA_VISIBLE_DEVICES=4 python test.py \
./logs/CTgraphEnvGoalChange-v0/nm_cavia/5_seed\=102fo\=Falselr\=0.5tau\=1.0_21_01_2021_23_43_49_cavia/config.json \
./saves/CTgraphEnvGoalChange-v0/nm_cavia/5_seed\=102fo\=Falselr\=0.5tau\=1.0_21_01_2021_23_43_49_cavia/policy-250.pt \
--env-config-path ./logs/CTgraphEnvGoalChange-v0/nm_cavia/5_seed\=102fo\=Falselr\=0.5tau\=1.0_21_01_2021_23_43_49_cavia/ctgraph.json
# final (iter 500)
CUDA_VISIBLE_DEVICES=4 python test.py \
./logs/CTgraphEnvGoalChange-v0/nm_cavia/5_seed\=102fo\=Falselr\=0.5tau\=1.0_21_01_2021_23_43_49_cavia/config.json \
./saves/CTgraphEnvGoalChange-v0/nm_cavia/5_seed\=102fo\=Falselr\=0.5tau\=1.0_21_01_2021_23_43_49_cavia/policy-500.pt \
--env-config-path ./logs/CTgraphEnvGoalChange-v0/nm_cavia/5_seed\=102fo\=Falselr\=0.5tau\=1.0_21_01_2021_23_43_49_cavia/ctgraph.json

# seed 42
# best (iter 500)
CUDA_VISIBLE_DEVICES=4 python test.py \
./logs/CTgraphEnvGoalChange-v0/nm_cavia/5_seed\=42fo\=Falselr\=0.5tau\=1.0_22_01_2021_02_21_28_cavia/config.json \
./saves/CTgraphEnvGoalChange-v0/nm_cavia/5_seed\=42fo\=Falselr\=0.5tau\=1.0_22_01_2021_02_21_28_cavia/policy-500.pt \
--env-config-path ./logs/CTgraphEnvGoalChange-v0/nm_cavia/5_seed\=42fo\=Falselr\=0.5tau\=1.0_22_01_2021_02_21_28_cavia/ctgraph.json
# final (iter 500)
CUDA_VISIBLE_DEVICES=4 python test.py \
./logs/CTgraphEnvGoalChange-v0/nm_cavia/5_seed\=42fo\=Falselr\=0.5tau\=1.0_22_01_2021_02_21_28_cavia/config.json \
./saves/CTgraphEnvGoalChange-v0/nm_cavia/5_seed\=42fo\=Falselr\=0.5tau\=1.0_22_01_2021_02_21_28_cavia/policy-500.pt \
--env-config-path ./logs/CTgraphEnvGoalChange-v0/nm_cavia/5_seed\=42fo\=Falselr\=0.5tau\=1.0_22_01_2021_02_21_28_cavia/ctgraph.json

# seed 378
# best (iter 500)
CUDA_VISIBLE_DEVICES=4 python test.py \
./logs/CTgraphEnvGoalChange-v0/nm_cavia/5_seed\=378fo\=Falselr\=0.5tau\=1.0_22_01_2021_04_27_43_cavia/config.json \
./saves/CTgraphEnvGoalChange-v0/nm_cavia/5_seed\=378fo\=Falselr\=0.5tau\=1.0_22_01_2021_04_27_43_cavia/policy-500.pt \
--env-config-path ./logs/CTgraphEnvGoalChange-v0/nm_cavia/5_seed\=378fo\=Falselr\=0.5tau\=1.0_22_01_2021_04_27_43_cavia/ctgraph.json
# final (iter 500)
CUDA_VISIBLE_DEVICES=4 python test.py \
./logs/CTgraphEnvGoalChange-v0/nm_cavia/5_seed\=378fo\=Falselr\=0.5tau\=1.0_22_01_2021_04_27_43_cavia/config.json \
./saves/CTgraphEnvGoalChange-v0/nm_cavia/5_seed\=378fo\=Falselr\=0.5tau\=1.0_22_01_2021_04_27_43_cavia/policy-500.pt \
--env-config-path ./logs/CTgraphEnvGoalChange-v0/nm_cavia/5_seed\=378fo\=Falselr\=0.5tau\=1.0_22_01_2021_04_27_43_cavia/ctgraph.json
