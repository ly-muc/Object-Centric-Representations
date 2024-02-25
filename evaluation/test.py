import argparse
import pathlib


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
        '--data_path', default='/Code/EPFL/VITA/causaltriplet/src/data')
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

    checkpoint_path = pathlib.Path(args.checkpoint_path)
    assert checkpoint_path.is_file(), "No checkpoint for evaluation found"


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
