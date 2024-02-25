import math
import matplotlib.pyplot as plt
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import pathlib

from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.utils as vutils

from tools.dataset import Procthor, CLEVR, create_dataset
from tools.metrics import get_onehot, adjusted_rand_index, mean_iou
from tools import loss_function, visualization, scheduler, ada_gvae
from slot_attention import SlotAutoEncoder
import wandb


def parse_arguments():
    """Experiment arguments.
    """
    parser = argparse.ArgumentParser(description='Experiment arguments')

    # Environment & Paths
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number cpu workers for fetching images')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument(
        '--checkpoint_path', default='checkpoint.pt.tar', help='Path to checkpoint file')
    parser.add_argument('--run_id', default=None,
                        help='Weights&Biases Experiment Run ID')
    parser.add_argument('--log_path', default='logs',
                        help='Path to logs folder')
    parser.add_argument(
        '--data_path', default='/home/linyan/Code/EPFL/VITA/causaltriplet/src/data')
    parser.add_argument('--project_name')
    parser.add_argument('--debug', type=bool, default=False)

    # Hyperparameter
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--clip', type=float, default=1.0,
                        help='Limit for gradient clipping')

    # Dataset
    parser.add_argument('--image_size', type=int, default=64,
                        help='Value that image width and height are resized to')
    parser.add_argument('--image_pair', type=bool,
                        default=False, help='Returns causal image tuple')
    parser.add_argument('--split', type=float, default=0.8,
                        help='80% Training images and 20% Validation')
    parser.add_argument('--task', default="procthor",
                        help='Options: [procthor, clevr]')
    parser.add_argument('--segmentation', default=True)
    parser.add_argument('--max_num_objects', type=int, default=15)

    return parser.parse_args()


def main(args):

    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available else "cpu"
    model = SlotAutoEncoder(device=device, implicit_diff=True, vae=False)

    filter_kwargs = {
        # 'perceptual_similarity': True,
        # 'perceptual_threshold':0.4,
        'num_objects': args.max_num_objects
    }

    color_map = visualization.ColorMap(args.max_num_objects)

    data_path = pathlib.Path(args.data_path)

    train_dataset = create_dataset(
        task_name=args.task,
        path_to_data=(data_path / 'train'),
        segmentation=args.segmentation,
        image_size=(args.image_size, args.image_size),
        image_pair=args.image_pair,
        file_ending="first",
        debug=args.debug,
        **filter_kwargs
        )
    val_dataset = create_dataset(
        task_name=args.task,
        path_to_data=(data_path / 'val'),
        segmentation=args.segmentation,
        image_size=(args.image_size, args.image_size),
        file_ending="first",
        debug=args.debug,
        **filter_kwargs
        )

    args.batch_size = args.batch_size if not args.image_pair else args.batch_size // 2

    train_loader_kwargs = {
        'batch_size': args.batch_size,
        'shuffle': True,
        'num_workers': args.num_workers,
        'pin_memory': True,
        'drop_last': True,
    }

    val_loader_kwargs = {
        'batch_size': args.batch_size,
        'shuffle': True,
        'num_workers': args.num_workers,
        'pin_memory': True,
        'drop_last': True,
    }

    # train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_len, val_len])
    train_loader = DataLoader(train_dataset, **train_loader_kwargs)
    val_loader = DataLoader(val_dataset, **val_loader_kwargs)

    start_epoch = 0
    epochs = args.epochs
    beta = 10.
    logging_step = 250 if not args.debug else 1

    lr_scheduler = scheduler.LinWarmupExpDecay(base_learning_rate=5e-4)

    checkpoint_path = pathlib.Path(args.checkpoint_path)
    load_checkpoint = checkpoint_path.is_file()

    # Replace with your own project and entity!
    if not load_checkpoint:
        checkpoint_path = pathlib.Path(
            args.log_path) / datetime.today().isoformat()
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        (checkpoint_path / "images").mkdir(parents=True, exist_ok=True)
        if not args.debug:
            wandb.init(project="", entity="")

    else:  # load checkpoint
        print("Beginn from checkpoint")
        checkpoint = torch.load(str(checkpoint_path), map_location='cpu')
        checkpoint_path = checkpoint_path.parent
        checkpoint_path = pathlib.Path("./")
        model.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch']
        lr_scheduler.global_step = torch.LongTensor(
            [checkpoint['global_step']])
        if not args.debug:
            wandb.init(project="", entity="")

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), 1e-4)

    if load_checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])

    for name, param in model.named_parameters():
        if "encoder" in name:
            param.requires_grad = False

    for epoch in range(start_epoch, epochs):

        # Training loop
        model.train()
        pbar = tqdm(train_loader)
        loss_agg = 0.
        for i, train_batch in enumerate(pbar):

            recon_loss = 0.
            contr_loss = 0.

            if args.segmentation:
                (segmentation, image) = train_batch
            else:
                image = train_batch

            if args.image_pair:
                img0 = image[0].to(device)
                img1 = image[1].to(device)

                # intial slot representation
                mu = model.slot_attention.slots_mu.expand(
                    args.batch_size, args.max_num_objects, -1)
                sigma = model.slot_attention.slots_logsigma.exp().expand(
                    args.batch_size, args.max_num_objects, -1)

                # slots shape [batch_size, num_slots, num_inputs]
                slot_init = mu + sigma * torch.randn(mu.shape, device=device)

                recon_combined, recons, mask, slots, _ = model(img0, slot_init)
                recon_combined_pair, _, _, slots_pair, _ = model(
                    img1, slot_init)

                reconstruction = (loss_function.reconstruction(
                    img0, recon_combined) + loss_function.reconstruction(img1, recon_combined_pair)) / 2
                consistency = loss_function.match_cosine_similarity(
                    slots, slots_pair)

                image = img0

                loss = reconstruction + beta * consistency

            else:
                image = image.to(device)
                recon_combined, recons, mask, slots, features = model(image)

                recon_loss = loss_function.reconstruction(
                    image, recon_combined)

                if beta > 0:
                    contr_loss = loss_function.contrastive_mask(mask, features)

                loss = recon_loss + beta * contr_loss

            loss.backward()
            loss_agg += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            optimizer.zero_grad()

            lr_scheduler.step()
            optimizer.param_groups[0]['lr'] = lr_scheduler.learning_rate
            
            pbar.set_description(
                f"Loss: {loss.item():.4f}, Reconstruction: {recon_loss.item():.4f},
                Contrastive: {contr_loss.item():.4f}, Learning rate: {lr_scheduler.learning_rate:.4f}")

            if i % logging_step == 0:

                # Create Visualization
                alpha_mask = color_map(mask.argmax(dim=1))
                slot_attn = (mask.softmax(dim=1) * recons) + \
                    (1 - mask.softmax(dim=1))
                visual = visualization.concatenate_images(
                    image, color_map(segmentation).to(device), recon_combined,
                    alpha_mask, slot_attn)
                grid = vutils.make_grid(
                    visual, nrow=args.max_num_objects+4, pad_value=0.2)[:, 2:-2, 2:-2]
                vutils.save_image(
                    grid, str(checkpoint_path / "images" / f'{epoch}_train_{i:04d}.png'))

                if not args.debug:
                    wandb.log({'Train/Loss': loss_agg / logging_step,
                              'Train/lr': lr_scheduler.learning_rate})
                    wb_image = wandb.Image(
                        grid, caption="Train/Reconstrued Image")
                    wandb.log({"Train/Reconstrued Image": wb_image})

                loss_agg = 0.

                checkpoint = {
                    'epoch': epoch,
                    'global_step': lr_scheduler.global_step,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }

                # torch.save(checkpoint, checkpoint_path / 'checkpoint.pt.tar')

        # Eval loop
        model.eval()
        with torch.no_grad():
            loss_agg = 0.
            ari_agg = 0.
            m_iou_agg = 0.

            for j, eval_batch in enumerate(val_loader):
                if args.segmentation:
                    (segmentation, image) = eval_batch
                else:
                    image = train_batch

                image = image.to(device)
                segmentation = segmentation.to(device)
                recon_combined, recons, mask, slots, _ = model(image)
                loss = torch.mean((image - recon_combined)**2)
                loss_agg += loss.item()

                # compute metrics
                if args.segmentation:
                    seg_oh, mask_oh = get_onehot(
                        segmentation, mask, args.batch_size, args.max_num_objects)
                    ari_agg += adjusted_rand_index(seg_oh, mask_oh,
                                                   n_points=args.image_size**2, mean=True)
                    m_iou_agg += mean_iou(seg_oh, mask_oh, mean=True)

                if j % logging_step == 0:

                    # Create Visualization
                    alpha_mask = color_map(mask.argmax(dim=1))
                    slot_attn = (mask.softmax(dim=1) * recons) + \
                        (1 - mask.softmax(dim=1))
                    visual = visualization.concatenate_images(
                        image, color_map(segmentation).to(
                            device), recon_combined,
                        alpha_mask, slot_attn)
                    grid = vutils.make_grid(
                        visual, nrow=args.max_num_objects+4, pad_value=0.2)[:, 2:-2, 2:-2]
                    vutils.save_image(
                        grid, str(checkpoint_path / "images" / f'{epoch:04d}.png'))

                    if not args.debug:
                        wb_image = wandb.Image(
                            grid, caption="Val/Reconstrued Image")
                        wandb.log({"Val/Reconstrued Image": wb_image})
                        wandb.log({'Val/loss': loss_agg/logging_step, 'Val/ari': (
                            ari_agg * 100)/logging_step, 'Val/m_iou': (m_iou_agg * 100)/logging_step})

                    print(f"====> Epoch {epoch+1}: Loss = {loss_agg/logging_step:.2f} \
                        \t ARI = {((ari_agg * 100)/logging_step):.2f} \
                        \t mIoU = {((m_iou_agg * 100)/logging_step):.2f}")

                    loss_agg = 0.
                    ari_agg = 0.
                    m_iou_agg = 0.


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
