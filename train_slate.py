import argparse
import os
import pathlib
from ast import parse
from datetime import datetime
from os import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.utils as vutils
import wandb
from dataset import CLEVR, Procthor
from metrics import adjusted_rand_index
from slate import SLATE
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
from torchvision.transforms import InterpolationMode, Resize
from train_utils import cosine_anneal, linear_warmup, visualize


def parse_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--patience', type=int, default=4)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--image_pair', type=bool, default=False)
    parser.add_argument('--task', default="procthor")

    parser.add_argument('--checkpoint_path', default='checkpoint.pt.tar')
    parser.add_argument('--run_id', default=None)
    parser.add_argument('--log_path', default='logs')
    parser.add_argument(
        '--data_path', default='/home/linyan/Code/EPFL/VITA/causaltriplet/src/data')

    parser.add_argument('--lr_dvae', type=float, default=3e-4)
    parser.add_argument('--lr_main', type=float, default=1e-4)
    parser.add_argument('--lr_warmup_steps', type=int, default=30000)

    parser.add_argument('--num_dec_blocks', type=int, default=4)
    parser.add_argument('--vocab_size', type=int, default=1024)
    parser.add_argument('--d_model', type=int, default=192)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)

    parser.add_argument('--num_iterations', type=int, default=3)
    parser.add_argument('--num_slots', type=int, default=5)
    parser.add_argument('--num_slot_heads', type=int, default=1)
    parser.add_argument('--slot_size', type=int, default=192)
    parser.add_argument('--mlp_hidden_size', type=int, default=192)
    parser.add_argument('--img_channels', type=int, default=3)
    parser.add_argument('--pos_channels', type=int, default=4)

    parser.add_argument('--tau_start', type=float, default=1.0)
    parser.add_argument('--tau_final', type=float, default=0.1)
    parser.add_argument('--tau_steps', type=int, default=30000)

    parser.add_argument('--max_num_objects', type=int, default=None)

    parser.add_argument('--hard', action='store_true')

    return parser.parse_args()


def main(args):

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Data loader
    resizer = Resize((args.image_size, args.image_size))
    def transform_image(x): return resizer(x) / 255.
    transform_segmentation = Resize(
        (args.image_size, args.image_size), interpolation=InterpolationMode.NEAREST)

    # Data filter arguments
    filter_kwargs = {
        # 'perceptual_similarity': True,
        # 'perceptual_threshold':0.4,
        'num_objects': args.max_num_objects
    }

    # Selected task
    if args.task == 'procthor':
        data = Procthor(
            path_to_data=args.data_path,
            transform_image=transform_image,
            transform_segmentation=transform_segmentation,
            segmentation=True,
            file_ending="first",
            **filter_kwargs
            )

    elif args.task == 'clevr':
        data = CLEVR(path_to_data=args.data_path,
                     transform_image=transform_image)

    else:
        raise ValueError("Invalid name for task type.")

    # Split into train and validation set
    train_len, val_len = int(len(data) * 0.8), len(data) - int(len(data) * 0.8)
    print(f"Initialized with {train_len} train files and {val_len} val files.")
    train_data, val_data = torch.utils.data.random_split(
        data, [train_len, val_len])

    # Create dataloader
    loader_kwargs = {
        'batch_size': args.batch_size,
        'shuffle': True,
        'num_workers': args.num_workers,
        'pin_memory': True,
        'drop_last': True,
    }
    train_loader = DataLoader(train_data, **loader_kwargs)
    val_loader = DataLoader(val_data, **loader_kwargs)

    train_epoch_size, val_epoch_size = len(train_loader), len(val_loader)

    # model
    model = SLATE(args)

    # training parameter
    log_interval = 5

    if os.path.isfile(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
        start_epoch = checkpoint['epoch']
        best_val_loss = checkpoint['best_val_loss']
        best_epoch = checkpoint['best_epoch']
        stagnation_counter = checkpoint['stagnation_counter']
        lr_decay_factor = checkpoint['lr_decay_factor']
        model.load_state_dict(checkpoint['model'])

    else:
        checkpoint = None
        start_epoch = 0
        best_val_loss = 100000
        best_epoch = 0
        stagnation_counter = 0
        lr_decay_factor = 1.0

    model.to(device)

    optimizer = torch.optim.Adam([
        {'params': (x[1] for x in model.named_parameters()
                    if 'dvae' in x[0]), 'lr': args.lr_dvae},
        {'params': (x[1] for x in model.named_parameters()
                    if 'dvae' not in x[0]), 'lr': args.lr_main},
    ])

    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
        log_dir = wandb.config["log_dir"]

    else:
        log_dir = os.path.join(args.log_path, datetime.today().isoformat())
        pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)

        wandb.config["log_dir"] = log_dir

    # begin training
    for epoch in range(start_epoch, args.epochs):

        model.train()

        for batch, train_batch in enumerate(train_loader):

            if args.task == 'procthor':
                (segmentation, image) = train_batch
            else:
                image = train_batch

            global_step = epoch * train_epoch_size + batch

            tau = cosine_anneal(
                global_step,
                args.tau_start,
                args.tau_final,
                0,
                args.tau_steps)

            lr_warmup_factor = linear_warmup(
                global_step,
                0.,
                1.0,
                0,
                args.lr_warmup_steps)

            optimizer.param_groups[0]['lr'] = lr_decay_factor * args.lr_dvae
            optimizer.param_groups[1]['lr'] = lr_decay_factor * \
                lr_warmup_factor * args.lr_main

            image = image.to(device)

            optimizer.zero_grad()

            (recon, cross_entropy, mse, attns) = model(image, tau, args.hard)

            loss = mse + cross_entropy

            loss.backward()
            clip_grad_norm_(model.parameters(), args.clip, 'inf')
            optimizer.step()

            with torch.no_grad():
                if batch % log_interval == 0:

                    if args.task == 'procthor':
                        # compute ari score
                        segmentation = segmentation.view(args.batch_size, -1)
                        slot_attns = torch.argmax(
                            attns, axis=1).view(args.batch_size, -1)
                        seg_oh = torch.nn.functional.one_hot(
                            segmentation.to(torch.int64), args.max_num_objects)
                        slot_oh = torch.nn.functional.one_hot(
                            slot_attns.to(torch.int64), args.max_num_objects)

                        train_ari = torch.mean(adjusted_rand_index(
                            seg_oh, slot_oh.to('cpu'), n_points=args.image_size**2)).item()

                    else:
                        train_ari = np.NaN

                    print('Train Epoch: {:3} [{:5}/{:5}] \t Loss: {:F} \t MSE: {:F} \t ARI = {:F}'.format(
                        epoch+1, batch, train_epoch_size, loss.item(), mse.item(), train_ari))

                    wandb.log({
                        'TRAIN/loss': loss.item(),
                        'TRAIN/ari': train_ari,
                        'TRAIN/cross_entropy': cross_entropy.item(),
                        'TRAIN/mse': mse.item(),
                        'TRAIN/tau': tau,
                        'TRAIN/lr_dvae': optimizer.param_groups[0]['lr'],
                        'TRAIN/lr_main': optimizer.param_groups[1]['lr']
                    })

        # save image
        with torch.no_grad():
            gen_img = model.reconstruct_autoregressive(image[:32])
            vis_attns = image.unsqueeze(
                1) * attns + 1. - attns  # visualize attention
            vis_recon = visualize(image, recon, gen_img, vis_attns, N=32)
            grid = vutils.make_grid(
                vis_recon, nrow=args.num_slots + 3, pad_value=0.2)[:, 2:-2, 2:-2]

            wb_image = wandb.Image(grid, caption="Reconstrued Image")
            wandb.log({"TRAIN/IMAGES": wb_image})

        with torch.no_grad():
            model.eval()

            val_cross_entropy_relax = 0.
            val_mse_relax = 0.

            val_cross_entropy = 0.
            val_mse = 0.

            val_ari_mean = 0.
            for batch, val_batch in enumerate(val_loader):

                if args.task == 'procthor':
                    (segmentation, image) = val_batch
                else:
                    image = val_batch

                image = image.to(device)

                (recon_relax, cross_entropy_relax, mse_relax,
                 attns_relax) = model(image, tau, False)

                (recon, cross_entropy, mse, attns) = model(image, tau, True)

                val_cross_entropy_relax += cross_entropy_relax.item()
                val_mse_relax += mse_relax.item()

                val_cross_entropy += cross_entropy.item()
                val_mse += mse.item()

                # compute ari score
                if args.task == 'procthor':
                    slot_attns = torch.argmax(attns, axis=1)
                    seg_oh = torch.nn.functional.one_hot(segmentation.view(
                        args.batch_size, -1).to(torch.int64), args.max_num_objects)
                    slot_oh = torch.nn.functional.one_hot(slot_attns.view(
                        args.batch_size, -1).to(torch.int64), args.max_num_objects)
                    val_ari = adjusted_rand_index(
                        seg_oh, slot_oh.to('cpu'), n_points=args.image_size**2)
                    val_ari_mean += torch.mean(val_ari)

            val_cross_entropy_relax /= (val_epoch_size)
            val_mse_relax /= (val_epoch_size)

            val_cross_entropy /= (val_epoch_size)
            val_mse /= (val_epoch_size)

            val_ari_mean /= (val_epoch_size)

            val_loss_relax = val_mse_relax + val_cross_entropy_relax
            val_loss = val_mse + val_cross_entropy

            wandb.log({
                'VAL/loss_relax': val_loss_relax,
                'VAL/cross_entropy_relax': val_cross_entropy_relax,
                'VAL/mse_relax': val_mse_relax,
                'VAL/loss': val_loss,
                'VAL/ari': val_ari_mean,
                'VAL/cross_entropy': val_cross_entropy,
                'VAL/mse': val_mse
            })

            print('====> Epoch: {:3} \t Loss = {:F} \t MSE = {:F} \t ARI = {:F}'.format(
                epoch+1, val_loss, val_mse, val_ari_mean))

            if val_loss < best_val_loss:
                stagnation_counter = 0
                best_val_loss = val_loss
                best_epoch = epoch + 1

                torch.save(model.state_dict(), os.path.join(
                    log_dir, 'best_model.pt'))

                if 5 <= epoch:
                    gen_img = model.reconstruct_autoregressive(image)
                    vis_attns = image.unsqueeze(
                        1) * attns + 1. - attns  # visualize attention
                    vis_recon = visualize(
                        image, recon, gen_img, vis_attns, N=32)
                    grid = vutils.make_grid(
                        vis_recon, nrow=args.num_slots + 3, pad_value=0.2)[:, 2:-2, 2:-2]
                    wb_image = wandb.Image(grid, caption="Reconstrued Image")
                    wandb.log({"VAL/IMAGES": wb_image})

                if args.task == 'procthor':
                    n_images = 16 if args.batch_size > 16 else args.batch_size
                    wb_list = list()
                    for i in range(n_images):
                        wb_mask = wandb.Image((image[i]+1)/2, caption=f'ARI: {val_ari[i]:.3f}', masks={
                            "prediction": {"mask_data": slot_attns[i].to('cpu').squeeze(0).numpy()},
                            "ground truth": {"mask_data": segmentation[i].squeeze(0).numpy()}})

                        wb_list.append(wb_mask)

                    wandb.log({"predictions": wb_list})

            checkpoint = {
                'epoch': epoch + 1,
                'best_val_loss': best_val_loss,
                'best_epoch': best_epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'stagnation_counter': stagnation_counter,
                'lr_decay_factor': lr_decay_factor,
            }

        torch.save(checkpoint, os.path.join(log_dir, 'checkpoint.pt.tar'))


if __name__ == "__main__":
    args = parse_arguments()

    # Replace with your own project and entity!
    if os.path.isfile(args.checkpoint_path):
        wandb.init(project="", resume="must",
                   id=args.run_id, entity="")
        wandb.config.update(args, allow_val_change=True)
        print("Resumed training.")
    else:
        wandb.init(project="", entity="")

    main(args)
