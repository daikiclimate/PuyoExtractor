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
# from evaluator import evaluator
from models import build_model


def get_arg():
    parser = argparse.ArgumentParser(description="image model for action segmentation")
    parser.add_argument("config", type=str, help="path of a config file")
    args = parser.parse_args()
    config = Dict(yaml.safe_load(open(args.config)))
    return config


def main():
    config = get_arg()
    device = config.device
    # test_set = return_data.return_testloader()
    _, test_set = return_data.return_dataloader(config)
    model = build_model.build_model(config)
    model = model.to(device)
    weight = torch.load(f"{config.save_folder}/model.pth")
    model.load_state_dict(weight)
    model.eval()

    preds = []
    pbar = tqdm.tqdm(total=len(test_set))
    for _, data1, data2 in test_set:
        gt_index = check_gt_index(data1, data2)

        data2 = data2[0].to(device)
        with torch.no_grad():
            output = model(data2)
        print(output)
        print(output[gt_index])
        max_index = torch.argmax(output)
        best_field_onehot = data2[max_index]
        best_field = torch.argmax(best_field_onehot, 0)

        player1 = data1.detach().cpu().reshape(-1)
        player2 = best_field.detach().cpu().reshape(-1)
        tf = player1 == player2
        tf = torch.all(tf)
        preds.append(tf)
        if len(preds) == 10:
            break

        pbar.update(1)


def check_gt_index(d1, d2):
    d1 = d1.reshape(-1)
    d2 = torch.argmax(d2, 2)[0]
    for i, d in enumerate(d2):
        if torch.all(d.reshape(-1) == d1):
            return i
    return 0


if __name__ == "__main__":
    main()
