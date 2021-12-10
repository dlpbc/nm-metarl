CUDA_VISIBLE_DEVICES=4 python main.py --env-name ML45-v2 \
--num-context-params 100 --hidden-size 250 --num-layers 2 --nonlinearity relu \
--fast-batch-size 10 --fast-lr 10.0 --meta-batch-size 45 --test-batch-size 20 --num-test-steps 4 \
--num-workers 4 --num-batches 501 --test-freq 10 --seed 42 \
--expname cavia
