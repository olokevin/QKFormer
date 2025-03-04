# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL

from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import numpy as np
import torch
import torchvision
from torch.utils.data import Subset
from robustbench.data import load_cifar10c
from torch.utils.data import Dataset
from PIL import Image

class CIFAR10CDataset(Dataset):
    """
    CIFAR-10-C Dataset that mimics the behavior of the CIFAR10 dataset in torchvision
    to be fully compatible with timm's data pipeline.
    """
    def __init__(self, images, labels, is_training=False, **kwargs):
        # Store original data
        if isinstance(images, torch.Tensor):
            if images.shape[1] == 3:  # If NCHW format
                images = images.permute(0, 2, 3, 1)  # Convert to NHWC for PIL compatibility
            self.data = images.cpu().numpy()
        else:
            self.data = images
            
        if isinstance(labels, torch.Tensor):
            self.targets = labels.cpu().numpy()
        else:
            self.targets = labels
        
        # Ensure data is in uint8 format for PIL conversion
        if self.data.dtype != np.uint8:
            if self.data.max() <= 1.0:
                self.data = (self.data * 255).astype(np.uint8)
            else:
                self.data = self.data.astype(np.uint8)
        
        # Standard CIFAR-10 classes
        self.classes = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        # Create the dataset_config that timm expects
        self.dataset_config = {
            'input_size': (3, 32, 32),
            'interpolation': 'bilinear',
            'mean': (0.4914, 0.4822, 0.4465),
            'std': (0.2470, 0.2435, 0.2616),
            'crop_pct': 1.0,
            'num_classes': len(self.classes)
        }
        
        # Store all the attributes timm expects
        self.transform = None  # Will be set by timm's create_loader
        self.target_transform = None
        self.is_training = is_training
        
        # Add additional attributes for timm compatibility
        for key, value in kwargs.items():
            setattr(self, key, value)
            
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        
        # Convert numpy to PIL Image
        img = Image.fromarray(img)
        
        # Apply transforms if they exist (will be set by timm's create_loader)
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        return img, target
    
    def __len__(self):
        return len(self.data)

def build_cifar10c_dataset(args):
    """
    Build CIFAR-10-C dataset with the same format as timm's create_dataset.
    The returned datasets can be directly passed to timm's create_loader.
    
    Args:
        args: Arguments containing dataset parameters
            - corruption_type: Type of corruption to use
            - train_n: Number of training examples
            - severity: Corruption severity level (1-5)
            - data_root: Directory containing CIFAR-10-C files
    
    Returns:
        train, validation, and test datasets compatible with timm's create_loader
    """
    corruption_type = args.corruption_type
    train_n = args.train_n
    severity = args.severity
    
    # Use the RobustBench's load_cifar10c function
    x_corr, y_corr = load_cifar10c(
        10000, severity, os.path.expanduser(args.data_root), False, [corruption_type]
    )

    # Split data into train, validation, and test sets
    labels = {}
    num_classes = int(max(y_corr)) + 1
    for i in range(num_classes):
        labels[i] = [ind for ind, n in enumerate(y_corr) if n == i]
    
    num_ex = train_n // num_classes
    tr_idxs = []
    val_idxs = []
    test_idxs = []
    
    for i in range(len(labels.keys())):
        np.random.shuffle(labels[i])
        tr_idxs.append(labels[i][:num_ex])
        val_idxs.append(labels[i][num_ex:num_ex+10])
        test_idxs.append(labels[i][num_ex+10:num_ex+100])
    
    tr_idxs = np.concatenate(tr_idxs)
    val_idxs = np.concatenate(val_idxs)
    test_idxs = np.concatenate(test_idxs)
    
    # Create datasets compatible with timm's create_loader
    dataset_train = CIFAR10CDataset(
        x_corr[tr_idxs], 
        y_corr[tr_idxs], 
        is_training=True,
        # Additional timm expected attributes
        split='train',
        root=args.data_root,
        batch_size=getattr(args, 'batch_size', 128),
        repeats=getattr(args, 'epoch_repeats', 0)
    )
    
    dataset_val = CIFAR10CDataset(
        x_corr[val_idxs], 
        y_corr[val_idxs], 
        is_training=False,
        # Additional timm expected attributes
        split='val',
        root=args.data_root,
        batch_size=getattr(args, 'batch_size', 128)
    )
    
    dataset_test = CIFAR10CDataset(
        x_corr[test_idxs], 
        y_corr[test_idxs], 
        is_training=False,
        # Additional timm expected attributes
        split='test',
        root=args.data_root,
        batch_size=getattr(args, 'batch_size', 128)
    )
    
    return dataset_train, dataset_val, dataset_test

class CustomVisionDataset(torchvision.datasets.VisionDataset):
    def __init__(self, tensor_dataset, transform=None, target_transform=None):
        super(CustomVisionDataset, self).__init__(root=None, transform=transform, target_transform=target_transform)
        self.tensor_dataset = tensor_dataset

    def __len__(self):
        return len(self.tensor_dataset)

    def __getitem__(self, index):
        sample, target = self.tensor_dataset[index]
        
        if self.transform is not None:
            sample = self.transform(sample)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        return sample, target

def build_imagenetc_dataset(args):
    corruption_type = args.corruption_type
    train_n = args.train_n
    severity = args.severity
    
    data_root = args.data_root
    image_dir = os.path.join(data_root, 'imagenet-c', corruption_type, str(severity))
    # dataset = ImageFolder(image_dir, transform=transforms.ToTensor())
    dataset = datasets.ImageFolder(image_dir)
    indices = list(range(len(dataset.imgs))) #50k examples --> 50 per class
    assert train_n <= 20000
    labels = {}
    y_corr = dataset.targets
    for i in range(max(y_corr)+1):
        labels[i] = [ind for ind, n in enumerate(y_corr) if n == i] 
    num_ex = train_n // (max(y_corr)+1)
    tr_idxs = []
    val_idxs = []
    test_idxs = []
    for i in range(len(labels.keys())):
        np.random.shuffle(labels[i])
        tr_idxs.append(labels[i][:num_ex])
        val_idxs.append(labels[i][num_ex:num_ex+10])
        # tr_idxs.append(labels[i][:num_ex+10])
        test_idxs.append(labels[i][num_ex+10:num_ex+20])
    tr_idxs = np.concatenate(tr_idxs)
    val_idxs = np.concatenate(val_idxs)
    test_idxs = np.concatenate(test_idxs)

    dataset_train = CustomVisionDataset(Subset(dataset, tr_idxs), transform=build_transform(True, args))
    dataset_val = CustomVisionDataset(Subset(dataset, val_idxs), transform=build_transform(False, args))
    dataset_test = CustomVisionDataset(Subset(dataset, test_idxs), transform=build_transform(False, args))
    
    return dataset_train, dataset_val, dataset_test


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    dataset = datasets.ImageFolder(root, transform=transform)

    print(dataset)

    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
