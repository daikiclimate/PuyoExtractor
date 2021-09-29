import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import wandb
import yaml
from addict import Dict

from build_loss import build_loss_func
from dataset import return_data
# from evaluator import evaluator
from models import build_model

# from sklearn.metrics import accuracy_score


SEED = 14
torch.manual_seed(SEED)


def get_arg():
    parser = argparse.ArgumentParser(description="image model for action segmentation")
    parser.add_argument("config", type=str, help="path of a config file")
    args = parser.parse_args()
    config = Dict(yaml.safe_load(open(args.config)))
    return config


def sweep(path):
    config = dict(yaml.safe_load(open(path)))
    sweep_id = wandb.sweep(config, project="ActionPurposeSegmentation")
    wandb.agent(sweep_id, main)


def main(sweep=False):
    config = get_arg()

    # config.save_folder = os.path.join(config.save_folder, config.model)
    if not os.path.exists(config.save_folder):
        os.makedirs(config.save_folder)

    if config.wandb:
        name = config.model + "_" + config.head
        wandb.init(project="ActionPurposeSegmentation", config=config, name=name)
        config = wandb.config

    device = config.device
    # torch.set_default_tensor_type("torch.cuda.FloatTensor")
    print("device:", device)
    # train_set, test_set = return_data.return_dataset(config)
    train_set, test_set = return_data.return_dataloader(config)

    model = build_model.build_model(config)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    criterion = build_loss_func(config)

    best_eval = 0
    pbar = tqdm.tqdm(total=config.epochs)
    for epoch in range(1, 1 + config.epochs):
        print("\nepoch:", epoch)

        t0 = time.time()
        train_loss = train(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            dataset=train_set,
            config=config,
            device=device,
        )
        print(f"\nlr: {scheduler.get_last_lr()}")
        t1 = time.time()
        print(f"\ntraining time :{round(t1 - t0)} sec")

        best_eval = test(
            model=model,
            dataset=test_set,
            config=config,
            device=device,
            best_eval=best_eval,
        )
        # if config.wandb:
        #     wandb.log(
        #         {
        #             "epoch": epoch,
        #             "loss": train_loss,
        #             "acc": best_eval,
        #             "lr": scheduler.get_last_lr(),
        #         }
        #     )
        scheduler.step()
        pbar.update(1)

    torch.save(model.state_dict(), f"{config.save_folder}/model.pth")


def train(
    model,
    optimizer,
    criterion,
    dataset,
    config,
    device,
):
    model.train()
    total_loss = 0
    counter = 0
    total_batch = len(dataset) * config.repeat
    t0 = time.time()
    for _ in range(config.repeat):
        for base, data1, data2 in dataset:
            t1 = time.time()
            load_time = t1 - t0
            #data1 = data1.to(device)
            #data2 = data2.to(device)

            d = torch.cat([base, data1, data2])
            d = d.to(device)
            out = model(d)
            base = out[:config.batch_size]
            data1_out = out[config.batch_size:config.batch_size*2]
            data2_out = out[config.batch_size*2:]

            # data1_out = model(data1)
            # data2_out = model(data2)
            #bs = data1_out.shape
            #diff = torch.ones(bs).float().to(device)
            #pred_diff = data1_out - data2_out
            # loss = criterion(pred_diff, diff)
            # loss = criterion(data1_out, data2_out, base)
            loss = criterion(base, data1_out, data2_out)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            counter += 1
            print(
                f"\rloss[{loss.item()}] load[{t1-t0:.3f}] counter[{counter}/{total_batch}]",
                end="",
            )
            t0 = time.time()

    #for i in range(10):
    #    print(data1_out[i], data2_out[i])
    print("")
    print(f"\rtotal_loss: [{total_loss / counter}]", end="")
    return total_loss / counter


def test(model, dataset, config, device, best_eval=0, th=0.6):
    model.eval()
    labels = []
    for _, data1, data2 in dataset:

        data2 = data2[0].to(device)
        with torch.no_grad():
            output = model(data2)
        max_index = torch.argmax(output)
        best_field_onehot = data2[max_index]
        best_field = torch.argmax(best_field_onehot, 0)

        player1 = data1.detach().cpu().reshape(-1)
        player2 = best_field.detach().cpu().reshape(-1)
        tf = player1 == player2
        tf = torch.all(tf)
        labels.append(tf)
        if len(labels) == 1000:
            break

    score = np.sum(labels) / len(labels)

    # labels, preds = np.array(labels), np.array(preds)
    # score = accuracy_score(labels, preds)

    # with open("test.pkl", "wb") as f:
    # pickle.dump(preds, f)
    # eval.set_data(labels, preds)
    # eval.print_eval(["accuracy"])
    # score = eval.return_eval_score()
    # score = np.mean(labels == preds)
    print(f"score [{score}]")
    if score > best_eval and score > th:
        path = os.path.join(
            config.save_folder,
            config.model
            + "_"
            + config.type
            + "_"
            + str(score).replace(".", "")
            + ".pth",
        )
        # torch.save(model.state_dict(), path)
    return max(score, best_eval)


if __name__ == "__main__":
    path = "config/config_tcn_sweep.yaml"
    main()
    # sweep(path)
