### imagenet
CUDA_VISIBLE_DEVICES=0 python train.py -c ZO.yml
CUDA_VISIBLE_DEVICES=0 nohup python train.py -c ZO.yml >/dev/null 2>&1 &
