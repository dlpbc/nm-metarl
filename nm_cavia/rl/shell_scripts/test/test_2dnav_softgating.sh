# NMCAVIA (softgating)
# seed 42
# final (iter 500)
CUDA_VISIBLE_DEVICES=3 python test.py \
logs/2DNavigation-v0/nm_cavia/5_seed\=42fo\=Falselr\=0.5tau\=1.0_20_05_2021_13_16_19_nmcavia-softgating/config.json \
saves/2DNavigation-v0/nm_cavia/5_seed\=42fo\=Falselr\=0.5tau\=1.0_20_05_2021_13_16_19_nmcavia-softgating/policy-497.pt

# seed 102
# final (iter 500)
#CUDA_VISIBLE_DEVICES=3 python test.py \
#./logs/2DNavigation-v0/nm_cavia/5_seed\=102fo\=Falselr\=0.5tau\=1.0_18_05_2021_11_54_41_nmcavia-softgating/config.json \
#./saves/2DNavigation-v0/nm_cavia/5_seed\=102fo\=Falselr\=0.5tau\=1.0_18_05_2021_11_54_41_nmcavia-softgating/policy-500.pt
# seed 378
# final (iter 500)
#CUDA_VISIBLE_DEVICES=3 python test.py \
#./logs/2DNavigation-v0/nm_cavia/5_seed\=378fo\=Falselr\=0.5tau\=1.0_19_05_2021_00_19_08_nmcavia-softgating/config.json \
#./saves/2DNavigation-v0/nm_cavia/5_seed\=378fo\=Falselr\=0.5tau\=1.0_19_05_2021_00_19_08_nmcavia-softgating/policy-500.pt

