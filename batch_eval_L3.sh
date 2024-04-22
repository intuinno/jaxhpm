CUDA_VISIBLE_DEVICES=1 python eval.py --top-ctx 8 --model logs/jax_mmnist/exp-051-rev7-L3_20240417_140910/state_300.pt   --exp-name exp-051-rev7-L3-300  --batch-size 8 --num-examples 1000  --num-samples 3 --device cuda:0
CUDA_VISIBLE_DEVICES=1 python eval.py --top-ctx 8 --model logs/jax_mmnist/exp-051-rev7-L3_20240417_140910/state_200.pt   --exp-name exp-051-rev7-L3-200  --batch-size 8 --num-examples 1000  --num-samples 3 --device cuda:0
CUDA_VISIBLE_DEVICES=1 python eval.py --top-ctx 8 --model logs/jax_mmnist/exp-051-rev7-L3_20240417_140910/state_100.pt   --exp-name exp-051-rev7-L3-100  --batch-size 8 --num-examples 1000  --num-samples 3 --device cuda:0

