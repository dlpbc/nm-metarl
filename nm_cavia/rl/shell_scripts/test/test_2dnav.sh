# CAVIA
# seed 102
# final (iter 500)
CUDA_VISIBLE_DEVICES=3 python test.py \
./logs/2DNavigation-v0/cavia/5_seed\=102fo\=Falselr\=0.5tau\=1.0_25_01_2021_12_03_13_cavia/config.json \
./saves/2DNavigation-v0/cavia/5_seed\=102fo\=Falselr\=0.5tau\=1.0_25_01_2021_12_03_13_cavia/policy-500.pt
# seed 42
# final (iter 500)
CUDA_VISIBLE_DEVICES=3 python test.py \
./logs/2DNavigation-v0/cavia/5_seed\=42fo\=Falselr\=0.5tau\=1.0_24_01_2021_23_22_14_cavia/config.json \
./saves/2DNavigation-v0/cavia/5_seed\=42fo\=Falselr\=0.5tau\=1.0_24_01_2021_23_22_14_cavia/policy-500.pt
# seed 378
# final (iter 500)
CUDA_VISIBLE_DEVICES=3 python test.py \
./logs/2DNavigation-v0/cavia/5_seed\=378fo\=Falselr\=0.5tau\=1.0_26_01_2021_02_14_31_cavia/config.json \
./saves/2DNavigation-v0/cavia/5_seed\=378fo\=Falselr\=0.5tau\=1.0_26_01_2021_02_14_31_cavia/policy-500.pt

# NMCAVIA
# seed 102
# final (iter 500)
CUDA_VISIBLE_DEVICES=3 python test.py \
./logs/2DNavigation-v0/nm_cavia/5_seed\=102fo\=Falselr\=0.5tau\=1.0_23_01_2021_21_29_41_cavia/config.json \
./saves/2DNavigation-v0/nm_cavia/5_seed\=102fo\=Falselr\=0.5tau\=1.0_23_01_2021_21_29_41_cavia/policy-500.pt
# seed 42
# final (iter 500)
CUDA_VISIBLE_DEVICES=3 python test.py \
./logs/2DNavigation-v0/nm_cavia/5_seed\=42fo\=Falselr\=0.5tau\=1.0_23_01_2021_05_05_58_cavia/config.json \
./saves/2DNavigation-v0/nm_cavia/5_seed\=42fo\=Falselr\=0.5tau\=1.0_23_01_2021_05_05_58_cavia/policy-500.pt
# seed 378
# final (iter 500)
CUDA_VISIBLE_DEVICES=3 python test.py \
./logs/2DNavigation-v0/nm_cavia/5_seed\=378fo\=Falselr\=0.5tau\=1.0_24_01_2021_13_25_14_cavia/config.json \
./saves/2DNavigation-v0/nm_cavia/5_seed\=378fo\=Falselr\=0.5tau\=1.0_24_01_2021_13_25_14_cavia/policy-500.pt

