from pathlib import Path
import argparse
import json
import math
import os
import sys
import time
import numpy as np
from tqdm import tqdm
import torch
from torch import nn, optim
import pandas as pd

import unet
from augmentations import ct_transform, aug_transform
from datasets import LiverDataset
from utils import MetricLogger, t_vMF_dice_loss, dice_score


torch.manual_seed(1312)
np.random.seed(1312)


def get_arguments():
    parser = argparse.ArgumentParser(description="Train a UNet model", add_help=False)

    # Data
    parser.add_argument("--data-dir", type=Path, default="/path/to/datasets", required=True,
                        help='Path to the dataframes')

    # Model
    parser.add_argument("--channels_list", type=str, default="64,128,256,512,1024", required=False,
                        help='List of the number of channels for each block. Must be of length 5.')
    parser.add_argument("--num_classes", type=int, default=3, required=False,
                        help='Number of classes for the segmentation task')
    parser.add_argument("--n_channels", type=int, default=1, required=False,
                        help='Number of channels in the input image')
    # parser.add_argument("--post-process", type=bool, default=False, required=False,
    #                     help='Whether to apply a post-processing step to the predictions during training')

    # Checkpoints
    parser.add_argument("--exp-dir", type=Path, default="./exp",
                        help='Path to the experiment folder, where all logs/checkpoints will be stored')
    parser.add_argument("--save-freq", type=int, default=None,
                        help='Save a checkpoint every [save-freq] epochs')

    # Optim
    parser.add_argument("--epochs", type=int, default=20,
                        help='Number of epochs')
    parser.add_argument("--batch-size", type=int, default=16,
                        help='Batch size')
    parser.add_argument("--base-lr", type=float, default=0.2,
                        help='Base learning rate, effective learning after warmup is [base-lr] * [batch-size] / 256')
    parser.add_argument("--wd", type=float, default=1e-6,
                        help='Weight decay')
    parser.add_argument("--momentum", type=float, default=0.9,
                        help='Momentum')
    parser.add_argument("--lmbda", type=float, default=128.,
                        help="Upper bound for the class weights in the t-vMF loss.")

    # Running
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')

    return parser


def main(args):


    gpu = torch.device(args.device)

    args.exp_dir.mkdir(parents=True, exist_ok=True)
    stats_file = open(args.exp_dir / "stats.txt", "a", buffering=1)
    print(args)
    print(args, file=stats_file)
    print(" ".join(sys.argv))
    print(" ".join(sys.argv), file=stats_file)


    train_df = pd.read_csv(os.path.join(args.data_dir, "train_df.csv"))
    dataset_train = LiverDataset(train_df,ct_transform=ct_transform,aug_transform=aug_transform)
    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=True,
    )

    val_df = pd.read_csv(os.path.join(args.data_dir, "val_df.csv"))
    dataset_val = LiverDataset(val_df,ct_transform=ct_transform,aug_transform=aug_transform)
    loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=True,
    )

    channels_list = [int(x) for x in args.channels_list.split(",")]
    model = unet.UNet(args.n_channels, args.num_classes, channels_list).to(gpu)
    # model = torch.compile(model)
    
    optimizer = optim.SGD(model.parameters(),
            lr=args.base_lr, 
            momentum=args.momentum,
            weight_decay=args.wd)
    

    # Loss function
    criterion = t_vMF_dice_loss
    k_tensor = torch.zeros(args.num_classes).to(gpu)


    if (args.exp_dir / "model.pth").is_file():
        print("resuming from checkpoint")
        ckpt = torch.load(args.exp_dir / "model.pth", map_location="cpu")
        start_epoch = ckpt["epoch"]
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
    else:
        start_epoch = 0

    if (args.exp_dir / "model_val.pth").is_file():
        print("Found validation checkpoint")
        best_val_loss = ckpt["loss_val"]
    else:
        best_val_loss = np.inf
    

    start_time = time.time()
    for epoch in range(start_epoch, args.epochs):

        # Training and validation loops
        epoch_loss_train, lr = train_one_epoch(model, epoch,optimizer, criterion, loader_train,gpu,k_tensor)
        epoch_loss_val = validate_one_epoch(model, epoch, criterion, loader_val, gpu,k_tensor,args.lmbda)


        # Save checkpoint
        state = dict(
            epoch=epoch + 1,
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            loss_train = epoch_loss_train,
            loss_val = epoch_loss_val,
        )
        torch.save(state, args.exp_dir / "model.pth")

        if args.save_freq is not None and (epoch + 1) % args.save_freq == 0:
            torch.save(state, args.exp_dir / f"model_{epoch}.pth")

        if epoch_loss_val < best_val_loss:
            print("Saving best validation checkpoint at epoch", epoch + 1, "with loss", epoch_loss_val,file=stats_file)
            best_val_loss = epoch_loss_val
            torch.save(state, args.exp_dir / "model_val.pth")

        print(json.dumps({
                "epoch": epoch,
                "loss_train": epoch_loss_train,
                "loss_val": epoch_loss_val,
                "lr": lr,
                "time": time.time() - start_time,
                }),
                file=stats_file)


def adjust_learning_rate(args, optimizer, loader, step,gpu):
    '''Warmup for the first 10 epochs, then cosine decay'''
    max_steps = args.epochs * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = args.base_lr * args.batch_size / 128
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def train_one_epoch(model, epoch,optimizer, criterion, loader, gpu, k_tensor):
    model.train()

    loss_epoch = 0
    dsc_epoch = torch.zeros(args.num_classes)
    batches_seen = 0

    pbar = tqdm(enumerate(loader, start=epoch * len(loader)), total=len(loader), desc=f"Epoch {epoch}")
    
    for step,(ct_scan, seg_scan, _) in pbar:
        x = ct_scan.to(gpu, non_blocking=True).float()
        y = seg_scan.to(gpu, non_blocking=True).float()

        lr = adjust_learning_rate(args, optimizer, loader, step,gpu)
        
        optimizer.zero_grad()

        output = model(x)
        loss = criterion(output, y,k_tensor)
        loss.backward()
        optimizer.step()

        # Logging
        output_argmax = output.argmax(dim=1)
        pred_seg = torch.nn.functional.one_hot(output_argmax, args.num_classes).permute(0, 3, 1, 2)
        dsc = dice_score(pred_seg, y)
        dsc_epoch += dsc
        loss_epoch += loss.item()
        batches_seen += 1

        pbar.set_postfix_str(f"Loss_batch {loss.item():.4f} - Loss_train_epoch {loss_epoch / batches_seen:.4f} - DSC_batch {dsc} - DSC_train_epoch {dsc_epoch / batches_seen}")

    return loss_epoch/batches_seen, lr


def validate_one_epoch(model, epoch, criterion, loader, gpu, k_tensor, lmbda):
    model.eval()

    loss_epoch = 0
    dsc_epoch = 0
    batches_seen = 0

    pbar = tqdm(enumerate(loader, start=epoch * len(loader)), total=len(loader), desc=f"Epoch {epoch}")
    with torch.no_grad():
        for step,(ct_scan, seg_scan,_) in pbar:
            x = ct_scan.to(gpu, non_blocking=True).float()
            y = seg_scan.to(gpu, non_blocking=True).float()

            output = model(x)
            loss = criterion(output, y, k_tensor).item()

            output_argmax = output.argmax(dim=1)
            pred_seg = torch.nn.functional.one_hot(output_argmax, args.num_classes).permute(0, 3, 1, 2)
            dsc = dice_score(pred_seg, y).item()

            k_tensor = (lmbda * dsc).to(gpu)

            loss_epoch += loss
            dsc_epoch += dsc
            batches_seen += 1

            pbar.set_postfix_str(f"Loss {loss.item():.4f} - Loss_val_epoch {loss_epoch / batches_seen:.4f}")

    return loss_epoch/batches_seen





if __name__ == "__main__":
    parser = argparse.ArgumentParser('UNet training script', parents=[get_arguments()])
    args = parser.parse_args()
    main(args)
