import argparse
import torch



def get_args():
    parser = argparse.ArgumentParser(description="<Project Name> Arguments", add_help=True)

    # Training Arguments
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs for training")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for the optimizer")
    parser.add_argument("--weight_decay", type=float, default=0.0001, help="Weight decay for the optimizer")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for the optimizer")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run the training on (cuda or cpu)")
    parser.add_argument("--save_ckpt_freq", type=int, default=10, help="Frequency of saving checkpoints")


    return parser.parse_args()


def main(args):
    pass


if __name__ == "__main__":
    opts = get_args()
    main(opts)
