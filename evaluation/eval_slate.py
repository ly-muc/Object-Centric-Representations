import argparse
import pathlib

import torch
from slate import SLATE
from torch.utils.data import DataLoader

from tools import visualization
from tools.dataset import create_dataset
from tools.metrics import adjusted_rand_index, get_onehot, mean_iou


def parse_arguments():
    """Experiment arguments.
    """
    parser = argparse.ArgumentParser(description='Experiment arguments')

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
    parser.add_argument('--data_path', default='/data/procthor')
    parser.add_argument('--segmentation', type=bool, default=True)

    parser.add_argument('--lr_dvae', type=float, default=3e-4)
    parser.add_argument('--lr_main', type=float, default=1e-4)
    parser.add_argument('--lr_warmup_steps', type=int, default=30000)

    parser.add_argument('--num_dec_blocks', type=int, default=4)
    parser.add_argument('--vocab_size', type=int, default=1024)
    parser.add_argument('--d_model', type=int, default=192)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)

    parser.add_argument('--num_iterations', type=int, default=3)
    parser.add_argument('--num_slots', type=int, default=15)
    parser.add_argument('--num_slot_heads', type=int, default=1)
    parser.add_argument('--slot_size', type=int, default=192)
    parser.add_argument('--mlp_hidden_size', type=int, default=192)
    parser.add_argument('--img_channels', type=int, default=3)
    parser.add_argument('--pos_channels', type=int, default=4)

    parser.add_argument('--tau_start', type=float, default=1.0)
    parser.add_argument('--tau_final', type=float, default=0.1)
    parser.add_argument('--tau_steps', type=int, default=30000)

    parser.add_argument('--max_num_objects', type=int, default=15)
    parser.add_argument('--hard', action='store_true')

    return parser.parse_args()


def main(args):

    torch.manual_seed(args.seed)
    torch.cuda.seed()

    device = "cuda" if torch.cuda.is_available else "cpu"
    model = SLATE(args)

    filter_kwargs = {
        # 'perceptual_similarity': True,
        # 'perceptual_threshold':0.4,
        'num_objects': args.max_num_objects
    }

    data_path = pathlib.Path(args.data_path)

    val_dataset = create_dataset(
        task_name=args.task,
        path_to_data=(data_path / 'val'),
        segmentation=args.segmentation,
        image_size=(args.image_size, args.image_size),
        file_ending="",
        **filter_kwargs
    )

    args.batch_size = \
        (args.batch_size if not args.image_pair else args.batch_size // 2)

    val_loader_kwargs = {
        'batch_size': args.batch_size,
        'shuffle': True,
        'num_workers': args.num_workers,
        'pin_memory': True,
        'drop_last': True,
    }

    val_loader = DataLoader(val_dataset, **val_loader_kwargs)

    checkpoint_path = pathlib.Path(args.checkpoint_path)
    assert checkpoint_path.is_file(), "No checkpoint for evaluation found"

    print("Beginn from checkpoint")
    checkpoint = torch.load(str(checkpoint_path), map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.to(device)

    # Eval loop
    model.eval()
    with torch.no_grad():

        ari_agg = list()
        m_iou_agg = list()

        for j, eval_batch in enumerate(val_loader):
            if args.segmentation:
                (segmentation, image) = eval_batch
            else:
                image = eval_batch

            image = image.to(device)
            segmentation = segmentation.to(device)

            (recon, cross_entropy, mse, attns, _) = model(image, 1, True)
            slot_attns = torch.argmax(attns, axis=1)

            # compute metrics
            if args.segmentation:
                seg_oh, mask_oh = get_onehot(
                    segmentation,
                    slot_attns,
                    args.batch_size,
                    args.max_num_objects
                )
                ari_agg.append(adjusted_rand_index(
                    seg_oh, mask_oh, n_points=args.image_size**2, mean=True))
                m_iou_agg.append(mean_iou(seg_oh, mask_oh, mean=True))

        ari_std, ari_mean = torch.std_mean(torch.Tensor(ari_agg))
        m_iou_std, m_iou_mean = torch.std_mean(torch.Tensor(m_iou_agg))

    print("Model performance")
    print(f"ARI [mean]: {ari_std} ARI [std]: {ari_mean}")
    print(f"mIoU [mean]: {m_iou_std} mIoU [std]: {m_iou_mean}")


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
