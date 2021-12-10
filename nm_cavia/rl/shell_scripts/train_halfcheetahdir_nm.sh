CUDA_VISIBLE_DEVICES=2 python main.py --env-name HalfCheetahDir-v1 \
--num-context-params 50 --hidden-size 200 --num-layers 2 --nonlinearity relu \
--fast-batch-size 20 --fast-lr 10.0 --meta-batch-size 40 --test-batch-size 40 --num-test-steps 4 \
--num-workers 4 --num-batches 501 --test-freq 10 --seed 378 \
--expname nmcavia-softgating --neuromodulation --nm-size 32 --nm-gate soft
