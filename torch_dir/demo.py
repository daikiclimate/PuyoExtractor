import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import tqdm
import yaml
from addict import Dict

from dataset import return_data
from dataset.puyo_dataset import PuyoGenerator, get_all_puyo_demo

# from evaluator import evaluator
from models import build_model
from dataset import transform


def get_arg():
    parser = argparse.ArgumentParser(description="image model for action segmentation")
    parser.add_argument("config", type=str, help="path of a config file")
    args = parser.parse_args()
    config = Dict(yaml.safe_load(open(args.config)))
    return config


def main():
    config = get_arg()
    device = config.device
    device = "cuda:3"
    # test_set = return_data.return_testloader()
    # _, test_set = return_data.return_dataloader(config)
    model = build_model.build_model(config)
    model = model.to(device)
    weight = torch.load(f"{config.save_folder}/model.pth")
    model.load_state_dict(weight)
    model.eval()

    field = np.full((12, 6), 6)

    generator = PuyoGenerator()

    n_put = 0
    max_put = 15
    _, test_transform = transform.return_transform()

    while True:
        next_puyo = generator.get_next_puyo()
        fields = get_all_puyo_demo(field.copy(), next_puyo, test_transform)
        fields = fields.to(device)

        with torch.no_grad():
            output = model(fields)
        max_index = torch.argmax(output)
        print(output.reshape(-1))

        best_field_onehot = fields[max_index]
        best_field = torch.argmax(best_field_onehot, 0)
        field = best_field.detach().cpu().numpy()
        print(field)

        if n_put == max_put:
            break
        n_put += 1
    exit()


def check_gt_index(d1, d2):
    d1 = d1.reshape(-1)
    d2 = torch.argmax(d2, 2)[0]
    for i, d in enumerate(d2):
        if torch.all(d.reshape(-1) == d1):
            return i
    return 0


if __name__ == "__main__":
    main()
