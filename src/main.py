import os
import sys
import argparse
import random
import math
import json
import time
import itertools
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from accelerate import Accelerator

from utils import redirect_stdout
from config import Config
from models.model_LRA import ModelForSC
from models.dataset_LRA import LRADataset






def print_summary(summary, save_if_improved, model, checkpoint_path):
    summary["loss"] = np.mean(summary["loss"])
    summary["accu"] = np.mean(summary["accu"])


    if summary["accu"] > summary["best_accu"]:
        summary["best_accu"] = summary["accu"]

    summary_round = {}
    for key in summary:
        if type(summary[key]) is str:
            summary_round[key] = summary[key]
        else:
            summary_round[key] = round(summary[key], 4)

    print(summary_round, flush = True)

    summary["t"] = 0
    summary["loss"] = []
    summary["accu"] = []

def step_LRA(model, optimizer, lr_scheduler, ds_iter, accelerator,
             init_t, summary, component, step_idx):

    t0 = time.time()
    optimizer.zero_grad()

    batch = next(ds_iter)
    batch_size = batch[list(batch.keys())[0]].size(0)

    if component == "train":
        with accelerator.accumulate(model):
            outputs = model(**batch)
            accelerator.backward(outputs["loss"].mean())
            accelerator.clip_grad_value_(model.parameters(), clip_value=1) # Gradient Clipping

            optimizer.step()
            lr_scheduler.step()
    else:
        with torch.no_grad(), accelerator.accumulate(model):
            outputs = model(**batch)

    t1 = time.time()

    t_escape = t1 - t0
    learning_rate = optimizer.param_groups[0]["lr"]
    loss = accelerator.gather_for_metrics(outputs["loss"]).mean().data.item()
    accu = accelerator.gather_for_metrics(outputs["accu"]).mean().data.item()
    time_since_start = time.time() - init_t

    if step_idx%100==0:
        accelerator.print(f"step={step_idx}, tt={time_since_start:.1f}, t={t_escape:.3f}, bs={batch_size}, lr={learning_rate:.6f}, loss={loss:.4f}, accu={accu:.4f}\t\t\t\t", end = "\r", flush = True)


    summary[component]["t"] += t_escape
    summary[component]["loss"].append(loss)
    summary[component]["accu"].append(accu)

    accelerator.log({"loss": loss, "accuracy": accu, "t": t_escape}, step=step_idx)
    return outputs

def train_LRA(model, optimizer, lr_scheduler, train_ds, dev_ds,
              accelerator, training_config, summary):

    checkpoint_path = training_config['checkpoint_path']
    # best_dev_loss = float(1e10)
    best_dev_accu = 0
    total_step = training_config["num_train_steps"]

    init_t = time.time()

    model.train()
    for train_step_idx in range(total_step):
        outputs = step_LRA(model, optimizer, lr_scheduler, train_ds, accelerator,
                           init_t, summary, component='train', step_idx=train_step_idx)

        if (train_step_idx + 1) % training_config["eval_frequency"] == 0:
            print_summary(summary["train"], False, model, checkpoint_path)
            model.eval()
            for dev_step_idx in range(training_config["num_eval_steps"]):
                outputs = step_LRA(model, optimizer, lr_scheduler, dev_ds, accelerator,
                                   init_t, summary, component='dev', step_idx=dev_step_idx)
            dev_accu = np.mean(summary["dev"]["accu"])
            if dev_accu > best_dev_accu:
                best_dev_accu = dev_accu
                if (train_step_idx + 1) > total_step * 0.2:
                    accelerator.wait_for_everyone()
                    accelerator.save({"model_state_dict":accelerator.get_state_dict(model)}, checkpoint_path)
                    accelerator.print('best model saved: step = ',train_step_idx, 'dev accu = ',dev_accu)

            print_summary(summary["dev"], True, model, checkpoint_path)
            model.train()


    accelerator.print('total training step (k): {}'.format(total_step/1000.0))
    accelerator.print("total training time (s): {}".format(int(time.time()-init_t)))
    accelerator.print("peak memory usage (MB): {}".format(torch.cuda.memory_stats()['active_bytes.all.peak']>>20))


def eval_LRA(model, optimizer, lr_scheduler, eval_ds, accelerator, summary, checkpoint_path): 
    init_t = time.time()
    model.eval()
    try:
        for test_step_idx in itertools.count():
            outputs = step_LRA(model, optimizer, lr_scheduler, eval_ds, accelerator,
                               init_t, summary, component='test', step_idx=test_step_idx)
    except StopIteration:
        print_summary(summary["test"], False, model, checkpoint_path)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type = str, default="train",
                        help="train eval")
    parser.add_argument("--checkpoint", type = str, default="test",
                        help="load ./checkpoints/model_name.model to evaluation")
    parser.add_argument("--attn", type = str, default="softmaxQKV",
                        help = "softmax, nystrom, linformer, informer, performer, bigbird, sketched, skeinb,skein, skein0, skeini, memorizing")
    parser.add_argument("--task", type = str, default="lra-listops",
                        help = "lra-listops, lra-retrieval, lra-text, lra-pathfinder32-curv_contour_length_14")
    parser.add_argument('--random', type=int, default=42)
    args = parser.parse_args()
    return args

def main():
    args = get_args()

    if args.task == 'lra-pathfinder':
        args.task = 'lra-pathfinder32-curv_contour_length_14'


    ### get model config ###
    model_config = Config[args.task]["model"]
    if args.attn in Config[args.task]["extra_attn_config"]:
        model_config.update(Config[args.task]["extra_attn_config"][args.attn])
    model_config["mixed_precision"] = True
    model_config["attn_type"] = args.attn
    model_config["max_seq_len"] = int(2 ** math.ceil(math.log2(model_config["max_seq_len"])))
    model_config["random_seed"] = args.random

    training_config = Config[args.task]["training"]

    ### log preparation ###
    log_dir = './log-{}/'.format(args.random)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    log_dir = os.path.join(log_dir, args.task)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    log_path = os.path.join(log_dir,'{}.{}.log'.format(args.mode, args.checkpoint))
    redirect_stdout(open(log_path, 'w'))
    summary = {
        component:{"t":0, "loss":[], "accu":[], "best_accu":0, "component":component}
        for component in ["train", "dev", "test"]
    }

    print(json.dumps([model_config, training_config], indent = 4))


    ###  set the random seeds for deterministic results. ####
    SEED = args.random
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True


    accumu_steps = model_config["bz_rate"] if "bz_rate" in model_config else 1
    mixed_precision = "fp16" if model_config.get("mixed_precision") else "no"
    print(f"Gradient Accumulation Steps: {accumu_steps}")
    print(f"Mixed Precision: {mixed_precision}")
    # device_ids = list(range(torch.cuda.device_count()))
    accelerator = Accelerator(gradient_accumulation_steps=accumu_steps,
                              mixed_precision=mixed_precision,
                              log_with=["tensorboard"],
                              logging_dir=os.path.join(log_dir,'{}.tensorboard'.format(args.checkpoint)))


    ### model preparation ###
    model = ModelForSC(model_config, batch_size=training_config["batch_size"], dual_input=(args.task == "lra-retrieval"))


    checkpoint_dir = './checkpoints-{}'.format(args.random)
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, '{}.{}.model'.format(args.checkpoint, args.random))
    training_config["checkpoint_path"] = checkpoint_path
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("model loaded from: " + checkpoint_path)

    model = accelerator.prepare(model)
    print(model)
    print(f"parameter_size: {[weight.size() for weight in model.parameters()]}", flush = True)
    print(f"num_parameter: {np.sum([np.prod(weight.size()) for weight in model.parameters()])}", flush = True)


    ### data preparation ###

    dataloaders = {
        "train":DataLoader(LRADataset(f"./data/lra_processed/{args.task}.train.pickle", True), batch_size = training_config["batch_size"], drop_last = True),
        "dev":DataLoader(LRADataset(f"./data/lra_processed/{args.task}.dev.pickle", True), batch_size = training_config["batch_size"], drop_last = True),
        "test":DataLoader(LRADataset(f"./data/lra_processed/{args.task}.test.pickle", False), batch_size = training_config["batch_size"], drop_last = True),
    }

    ### training preparation ###

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr = training_config["learning_rate"],
        betas = (0.9, 0.999), eps = 1e-6, weight_decay = training_config["weight_decay"]
    )
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer = optimizer,
        max_lr = training_config["learning_rate"],
        pct_start = training_config["warmup"] / training_config["num_train_steps"],
        anneal_strategy = training_config["lr_decay"],
        total_steps = training_config["num_train_steps"]
    )
    optimizer, dataloaders["train"], dataloaders["dev"], dataloaders["test"], lr_scheduler = accelerator.prepare(
        optimizer, dataloaders["train"], dataloaders["dev"], dataloaders["test"], lr_scheduler
    )
    data_iters = {k: iter(dataloader) for k, dataloader in dataloaders.items()}
    

    ### train ###
    if args.mode == 'train':
        accelerator.init_trackers(project_name=f"{args.task}-{args.attn}-{args.checkpoint}",
                                  config=training_config,
                                  init_kwargs={"run_name": f"run-{args.random}"})
        train_LRA(model, optimizer, lr_scheduler, data_iters["train"], data_iters["dev"], accelerator, training_config, summary)
        accelerator.end_training()

    ### eval ###
    if os.path.exists(checkpoint_path) and checkpoint_path != './checkpoints/test.model':
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("loading the best model from: " + checkpoint_path)
    eval_LRA(model, optimizer, lr_scheduler, data_iters["test"], accelerator, training_config, summary, checkpoint_path=training_config['checkpoint_path'])



if __name__ == '__main__':
    main()
