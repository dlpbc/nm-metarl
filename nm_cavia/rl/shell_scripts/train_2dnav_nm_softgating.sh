CUDA_VISIBLE_DEVICES=4 python main.py --env-name 2DNavigation-v0 \
--num-context-params 5 --hidden-size 100 --nonlinearity relu \
--fast-batch-size 20 --fast-lr 0.5 --meta-batch-size 20 --test-batch-size 40 --num-test-steps 4 \
--num-workers 4 --num-batches 501 --test-freq 10 --seed 42 \
--expname nmcavia-softgating --neuromodulation --nm-size 4 --nm-gate soft
