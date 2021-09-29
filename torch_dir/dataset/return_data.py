import torch
from torch.utils.data import DataLoader

from .puyo_dataset import PuyoDataset
from .transform import return_transform


def return_dataset(config):
    # dataset_type = config.type
    num_test = 2000
    transform = return_transform()
    return PuyoDataset(num_test=num_test, transform=transform), PuyoDataset(
        mode="test", num_test=num_test
    )


def return_dataloader(config):
    # train_set, valid_set = return_dataset(config, fold_num)
    train_set, valid_set = return_dataset(config)
    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        # num_workers=8,
        pin_memory=False,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_set,
        batch_size=1,
        drop_last=False,
        num_workers=0,
    )
    return train_loader, valid_loader


def return_testloader():
    transforms = return_test_img_transform()
    test_dataset = ImgDataset(mode="test", transform=transforms)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        drop_last=False,
        num_workers=0,
    )
    return test_loader
