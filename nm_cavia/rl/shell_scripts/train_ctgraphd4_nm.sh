CUDA_VISIBLE_DEVICES=4 python main.py --env-name CTgraphEnvGoalChange-v0 \
--num-context-params 20 --hidden-size 600 --nonlinearity relu \
--fast-batch-size 60 --fast-lr 0.5 --meta-batch-size 20 --test-batch-size 100 --num-test-steps 4 \
--num-workers 4 --num-batches 1501 --test-freq 10 --seed 378 \
--expname cavia --env-config-path ./envs/ctgraph/ctgraph_d4.json --neuromodulation --nm-size 32
