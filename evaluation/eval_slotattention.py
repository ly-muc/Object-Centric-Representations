import torch
import argparse
import pathlib

from torch.utils.data import DataLoader

from tqdm import tqdm

from tools.dataset import create_dataset
from tools.metrics import get_onehot, adjusted_rand_index, mean_iou
from slot_attention import SlotAutoEncoder


def parse_arguments():
    """Experiment arguments.
    """
    parser = argparse.ArgumentParser(description='Experiment arguments')

  # Environment & Paths
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number cpu workers for fetching images')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument(
        '--checkpoint_path',
        default='checkpoint.pt.tar',
        help='Path to checkpoint file')
    parser.add_argument('--run_id', default=None,
                        help='Weights&Biases Experiment Run ID')
    parser.add_argument('--log_path', default='logs',
                        help='Path to logs folder')
    parser.add_argument(
        '--data_path',
        default='Code/EPFL/VITA/causaltriplet/src/data')
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
    torch.cuda.seed()

    device = "cuda" if torch.cuda.is_available else "cpu"
    model = SlotAutoEncoder(device=device, implicit_diff=True, vae=False)

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
        args.batch_size if not args.image_pair else args.batch_size // 2

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

        for eval_batch in tqdm(val_loader):
            if args.segmentation:
                (segmentation, image) = eval_batch
            else:
                image = eval_batch

            image = image.to(device)
            segmentation = segmentation.to(device)

            recon_combined, recons, mask, _, _ = model(image)

            # compute metrics
            if args.segmentation:
                seg_oh, mask_oh = get_onehot(
                    segmentation, mask, args.batch_size, args.max_num_objects)
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
