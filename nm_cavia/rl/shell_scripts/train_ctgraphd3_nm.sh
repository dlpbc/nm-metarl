CUDA_VISIBLE_DEVICES=4 python main.py --env-name CTgraphEnvGoalChange-v0 \
--num-context-params 10 --hidden-size 300 --nonlinearity relu \
--fast-batch-size 25 --fast-lr 0.5 --meta-batch-size 40 --test-batch-size 40 --num-test-steps 4 \
--num-workers 4 --num-batches 1001 --test-freq 10 --seed 378 \
--expname cavia --env-config-path ./envs/ctgraph/ctgraph_d3.json --neuromodulation --nm-size 16
