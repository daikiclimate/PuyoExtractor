import math
import os
import pickle
import random

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
import tqdm


class PuyoDataset(data.Dataset):
    def __init__(
        self,
        mode="train",
        data_dir="/home/ubuntu/local/puyo_dataset",
        transform=None,
        num_test=2000,
    ):
        self._data_dir = data_dir
        self._puyo_list = os.listdir(data_dir)
        file_name = "./dataset/valid_files.pkl"
        if False:
            # if True:
            gt_labels, files = check_file_list(self._puyo_list)
            print(len(gt_labels))
            with open(file_name, "wb") as f:
                pickle.dump([gt_labels, files], f)
        else:
            with open(file_name, "rb") as f:
                t = pickle.load(f)
                gt_labels, files = t
            self._puyo_list = files

        self._mode = mode
        self._transform = transform
        if self._mode == "train":
            self._puyo_list = self._puyo_list[:-num_test]
        else:
            self._puyo_list = self._puyo_list[-num_test:]

    def __getitem__(self, idx):
        path = os.path.join(self._data_dir, self._puyo_list[idx])
        if self._mode == "test":
            pre_field, cur_field, exsample_field = get_all_puyo(
                path, transform=self._transform
            )
        else:
            pre_field, cur_field, exsample_field = puyo(path, self._transform)
        return pre_field, cur_field, exsample_field

    def __len__(self):
        return len(self._puyo_list)


def puyo(file_name="S103_10_p1_1.pkl", transform=None):

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

    pre_field = pre_field.reshape(1, 12, 6)
    cur_field = cur_field.reshape(1, 12, 6)
    exsample_field = exsample_field.reshape(1, 12, 6)

    pre_field, cur_field, exsample_field = transform(
        [pre_field, cur_field, exsample_field]
    )

    exsample_field = to_onehot(exsample_field).float()
    cur_field = to_onehot(cur_field).float()
    pre_field = to_onehot(pre_field).float()
    return pre_field, cur_field, exsample_field


def to_onehot(t):
    t = torch.tensor(t)
    puyo = t[0:1]
    puyo = puyo.long().reshape(-1)
    puyo = torch.eye(7)[puyo]
    puyo = puyo.reshape(12, 6, 7)
    puyo = puyo.permute(2, 0, 1)
    return torch.cat([puyo, t[1:]], 0)


def get_all_puyo(file_name, transform=None):
    with open(file_name, "rb") as f:
        t = pickle.load(f)
    pre_field, cur_field, next_puyo = t

    all_field = []
    for i in range(11):
        for j in range(2):
            f = put_next_puyo(pre_field, next_puyo[0], i, j)
            f = f.reshape(1, 12, 6)
            all_field.append(f)
    if transform:
        all_field = transform(all_field)

    all_field = [to_onehot(f).float().unsqueeze(0) for f in all_field]
    # all_field = [to_onehot(f).float().unsqueeze(0) for f in all_fields]
    all_field = torch.cat(all_field)
    return pre_field, cur_field, all_field


def put_next_puyo(field, puyo, put_place=None, flip_puyo=None):
    if flip_puyo == None:
        flip_puyo = random.randint(0, 1)
    if flip_puyo == 1:
        puyo[0], puyo[1] = puyo[1], puyo[0]

    tmp_field = field.copy()

    # put_place = random.randint(0, 5)
    # put_place = random.randint(6, 10)
    # put_place = 5 + 3
    if put_place == None:
        put_place = random.randint(0, 10)

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


def check_file_list(files, data_dir="/home/ubuntu/local/puyo_dataset"):
    gt_label = []
    valid_files = []
    for f in tqdm.tqdm(files):
        path = os.path.join(data_dir, f)
        pre_field, cur_field, exsample_field = get_all_puyo(path)
        gt_index = check_gt_index(cur_field, exsample_field)
        if gt_index == -1:
            continue
        else:
            gt_label.append(gt_index)
            valid_files.append(f)
    print(len(valid_files), len(files))
    return gt_label, valid_files


def check_gt_index(d1, d2):
    d1 = d1.reshape(-1)
    d1 = torch.tensor(d1)
    d2 = torch.argmax(d2, 1)
    for i, d in enumerate(d2):
        if torch.all(d.reshape(-1) == d1):
            return i
    return -1


def get_all_puyo_demo(pre_field, next_puyo, transform=None):

    all_field = []
    for i in range(11):
        for j in range(2):
            f = put_next_puyo(pre_field, next_puyo, i, j)
            f = f.reshape(1, 12, 6)
            all_field.append(f)
    if transform:
        all_field = transform(all_field)
    all_field = [to_onehot(f).float().unsqueeze(0) for f in all_field]
    all_field = torch.cat(all_field)
    return all_field


class PuyoGenerator(object):
    def __init__(self):
        self._use_puyo = np.random.permutation([0, 1, 2, 3, 4])[:4]

    def get_next_puyo(self):
        p1 = np.random.randint(0, 4)
        p2 = np.random.randint(0, 4)
        return self._use_puyo[[p1, p2]]


if __name__ == "__main__":
    a = PuyoDataset()
    for i in a:
        exit()
        print(i)
    # for i in range(1):
    #     puyo()
    # d= ImgDataset(mode="valid")
    # for img, label in d:
    #     print(label)
