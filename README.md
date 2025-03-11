# Fine-tuning Large Pre-trained SNNs with Just Forward Passes

Course project for W'25 ECE274

Yequan Zhao, Jiayi Tian, Liyan Tan, Yubin Deng

## Requirements

```
timm==0.6.12
cupy==11.4.0
torch==1.12.1
spikingjelly==0.0.0.0.12
pyyaml
tensorboard
```

## Dataset
CIFAR-10-C, CIFAR-100-C, ImageNet-C datasets will be automatically downloaded on the first run.

## Pretrained model
We have provided our pre-trained model on CIFAR-10 and CIFAR-100. For model pretrained on ImageNet, please download the 82.04 accuracy model from https://drive.google.com/drive/folders/1vhq9jmhmuyZ5_RGHuWD4wniza856qF8U?usp=drive_link and place under imagenet/ folder.

## Train & Test
### Table I
```
cd cifar10
bash table_1.sh
```

### Training on CIFAR-10-C (Table II, Table IV)
```
cd cifar10
bash run.sh
```

### Training on CIFAR-100-C (Table III, Table V)
```
cd cifar100
bash run.sh
```

## Acknowledgement & Contact Information
This code base is built upon [QKFormer](https://github.com/zhouchenlin2096/QKFormer), [spikingjelly](https://github.com/fangwei123456/spikingjelly).

