### imagenet
CUDA_VISIBLE_DEVICES=0 python train.py -c ZO.yml
CUDA_VISIBLE_DEVICES=0 nohup python train.py -c ZO.yml >/dev/null 2>&1 &

### cifar-10
# test
CUDA_VISIBLE_DEVICES=0 python train.py -c cifar10.yml --eval