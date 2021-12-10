CUDA_VISIBLE_DEVICES=3 python main.py --env-name ML45-v2 \
--num-context-params 100 --hidden-size 200 --num-layers 2 --nonlinearity relu \
--fast-batch-size 10 --fast-lr 10.0 --meta-batch-size 45 --test-batch-size 20 --num-test-steps 4 \
--num-workers 4 --num-batches 501 --test-freq 10 --seed 378 \
--expname nmcavia-softgating --neuromodulation --nm-size 32 --nm-gate soft
