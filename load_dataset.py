import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler

def create_torch_loaders(data_dir, batch_size):
    transform = transforms.Compose(
        [transforms.Resize(size=(460, 460)), transforms.ToTensor()]
    )

    dataset = datasets.ImageFolder(data_dir, transform=transform)

    indices = list(range(len(dataset)))
    validation = int(np.floor(0.85 * len(dataset)))

    print(0, validation, len(dataset))
    print(f"length of train size :{validation}")
    print(f"length of validation size :{len(dataset)-validation}")

    # indices = list(range(2000))
    # split = int(np.floor(0.85 * 2000))
    # validation = int(np.floor(0.70 * split))

    # print(0, validation, split, 2000)
    # print(f"length of train size :{validation}")
    # print(f"length of validation size :{split - validation}")
    # print(f"length of test size :{2000-split}")

    np.random.shuffle(indices)

    train_indices, validation_indices = (
        indices[:validation],
        indices[validation:],
    )

    train_sampler = SubsetRandomSampler(train_indices, generator=torch.Generator())
    validation_sampler = SubsetRandomSampler(validation_indices, generator=torch.Generator())

    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=train_sampler
    )

    validation_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=validation_sampler
    )

    return dataset, train_loader, validation_loader

def create_test_loader(data_dir, batch_size):
    transform = transforms.Compose(
        [transforms.Resize(255), transforms.CenterCrop(224), transforms.ToTensor()]
    )

    dataset = datasets.ImageFolder(data_dir, transform=transform)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    return test_loader