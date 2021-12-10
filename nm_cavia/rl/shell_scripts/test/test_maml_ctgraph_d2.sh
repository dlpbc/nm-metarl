# seed 42
# MAML
# best (itr 190)
python test.py \
./logs/CTgraphEnvGoalChange-v0/maml/seed\=42fo\=Falselr\=0.01tau\=1.0_11_02_2021_02_58_00_maml/config.json \
./saves/CTgraphEnvGoalChange-v0/maml/seed\=42fo\=Falselr\=0.01tau\=1.0_11_02_2021_02_58_00_maml/policy-190.pt \
--env-config-path ./logs/CTgraphEnvGoalChange-v0/maml/seed\=42fo\=Falselr\=0.01tau\=1.0_11_02_2021_02_58_00_maml/ctgraph.json

# best (itr 500)
python test.py \
./logs/CTgraphEnvGoalChange-v0/maml/seed\=42fo\=Falselr\=0.01tau\=1.0_11_02_2021_02_58_00_maml/config.json \
./saves/CTgraphEnvGoalChange-v0/maml/seed\=42fo\=Falselr\=0.01tau\=1.0_11_02_2021_02_58_00_maml/policy-500.pt \
--env-config-path ./logs/CTgraphEnvGoalChange-v0/maml/seed\=42fo\=Falselr\=0.01tau\=1.0_11_02_2021_02_58_00_maml/ctgraph.json

# NM_MAML
# best (itr 190)
python test.py \
./logs/CTgraphEnvGoalChange-v0/nm_maml/seed\=42fo\=Falselr\=0.01tau\=1.0_10_02_2021_18_14_01_maml/config.json \
./saves/CTgraphEnvGoalChange-v0/nm_maml/seed\=42fo\=Falselr\=0.01tau\=1.0_10_02_2021_18_14_01_maml/policy-190.pt \
--env-config-path ./logs/CTgraphEnvGoalChange-v0/nm_maml/seed\=42fo\=Falselr\=0.01tau\=1.0_10_02_2021_18_14_01_maml/ctgraph.json

# best (itr 500)
python test.py \
./logs/CTgraphEnvGoalChange-v0/nm_maml/seed\=42fo\=Falselr\=0.01tau\=1.0_10_02_2021_18_14_01_maml/config.json \
./saves/CTgraphEnvGoalChange-v0/nm_maml/seed\=42fo\=Falselr\=0.01tau\=1.0_10_02_2021_18_14_01_maml/policy-500.pt \
--env-config-path ./logs/CTgraphEnvGoalChange-v0/nm_maml/seed\=42fo\=Falselr\=0.01tau\=1.0_10_02_2021_18_14_01_maml/ctgraph.json
