import math
import os
import pickle
import random

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data


class PuyoDataset(data.Dataset):
    def __init__(
        self,
        mode="train",
        data_dir="./puyo_dataset",
        transform=None,
    ):
        self._data_dir = data_dir
        self._puyo_list = os.listdir(data_dir)
        # self._transform = transform
        # self._mode = mode

    def __getitem__(self, idx):
        path = os.path.join(self._data_dir, self._puyo_list[idx])
        cur_field, exsample_field = puyo(path)
        return cur_field, exsample_field

    def __len__(self):
        return len(self._puyo_list)


def puyo(file_name="S103_10_p1_1.pkl"):

    with open(file_name, "rb") as f:
        t = pickle.load(f)
    pre_field, cur_field, next_puyo = t
    exsample_field = put_next_puyo(pre_field, next_puyo[0])
    is_same = np.array_equal(exsample_field, cur_field)
    max_time = 0
    while is_same:
        is_same = np.array_equal(exsample_field, cur_field)
        exsample_field = put_next_puyo(pre_field, next_puyo[0])
        max_time += 1
        if max_time == 10:
            break
    # print(pre_field)
    # print(exsample_field)
    # print(cur_field)
    # print(exsample_field-cur_field)
    exsample_field = to_onehot(exsample_field).float()
    cur_field = to_onehot(cur_field).float()
    return cur_field, exsample_field


def to_onehot(t):
    t = torch.tensor(t).long().reshape(-1)
    t = torch.eye(7)[t]
    t = t.reshape(12, 6, 7)
    t = t.permute(2, 0, 1)
    return t


def put_next_puyo(field, puyo):
    flip_puyo = random.randint(0, 1)
    if flip_puyo:
        puyo[0], puyo[1] = puyo[1], puyo[0]

    tmp_field = field.copy()

    # put_place = random.randint(0, 5)
    # put_place = random.randint(6, 10)
    put_place = random.randint(0, 10)
    # put_place = 5 + 3
    blank_field = np.full((2, 6), 6, dtype=np.uint8)
    tmp_field = np.concatenate([blank_field, tmp_field])
    if put_place < 6:
        # tate
        # 0 <= put_place < 6
        tmp_field[0:2, put_place] = puyo
        drop_field(tmp_field, put_place, mode="tate")
    else:
        # yoko
        # 6 <= put_place <= 10
        # 0 <= put_place <= 4
        put_place -= 6
        tmp_field[0, put_place] = puyo[0]
        tmp_field[0, put_place + 1] = puyo[1]
        drop_field(tmp_field, put_place, mode="yoko")
    tmp_field = tmp_field[2:]
    return tmp_field


def drop_field(field, put_place, mode):
    if mode == "tate":
        for i in range(1, 13):
            if field[i, put_place] == 6:
                break
            elif field[i + 1, put_place] != 6:
                break
            else:
                field[i + 1, put_place] = field[i, put_place]
                field[i, put_place] = 6
        for i in range(0, 12):
            if field[i, put_place] == 6:
                break
            elif field[i + 1, put_place] != 6:
                break
            else:
                field[i + 1, put_place] = field[i, put_place]
                field[i, put_place] = 6
    if mode == "yoko":
        for i in range(0, 13):
            if field[i, put_place] == 6:
                break
            elif field[i + 1, put_place] != 6:
                break
            else:
                field[i + 1, put_place] = field[i, put_place]
                field[i, put_place] = 6
        for i in range(0, 13):
            if field[i, put_place + 1] == 6:
                break
            elif field[i + 1, put_place + 1] != 6:
                break
            else:
                field[i + 1, put_place + 1] = field[i, put_place + 1]
                field[i, put_place + 1] = 6


def get_list(path="puyo_dataset"):
    files = os.listdir(path)
    for i in range(len(files)):
        print(f"\r{i}:{files[i]}", end="")
        a = puyo(os.path.join(path, files[i]))


if __name__ == "__main__":
    a = PuyoDataset()
    for i in a:
        print(i)
        exit()
    # for i in range(1):
    #     puyo()
    # d= ImgDataset(mode="valid")
    # for img, label in d:
    #     print(label)
