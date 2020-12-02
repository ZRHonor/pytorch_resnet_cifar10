from torchvision.datasets import CIFAR10, CIFAR100
from torchvision import transforms
import numpy as np
def get_dataset(dataset, lt_factor):
    root_pth = 'data/'+dataset
    dataset_c = eval(dataset)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_dataset = dataset_c(root=root_pth, train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True)


    valid_dataset = dataset_c(root=root_pth, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))

    if lt_factor == 1: 
        return train_dataset, valid_dataset
    else:
        new_data = []
        new_targets = []
        if dataset == 'CIFAR10':
            num_examples = list(np.int32(np.linspace(1, 1/lt_factor, 10)*5000))
        else:
            num_examples = list(np.int32(np.linspace(1, 1/lt_factor, 100)*5000))
        counter = 0
        for cat_id in train_dataset.targets:
            if num_examples[cat_id] > 0:
                num_examples[cat_id] -= 1
            else:
                continue
            new_data.append(train_dataset.data[counter])
            new_targets.append(cat_id)
            counter += 1
        train_dataset.data = new_data
        train_dataset.targets = new_targets
        return train_dataset, valid_dataset

