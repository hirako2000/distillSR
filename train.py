import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from training.config import TrainingConfig
from training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train RealPLKSR")
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--pretrain', type=str, help='Path to pretrained weights')
    parser.add_argument('--device', type=str, help='Override device (cuda/mps/cpu)')

    args = parser.parse_args()
    config = TrainingConfig.from_yaml(args.config)

    if args.resume:
        config.resume_path = args.resume
    if args.pretrain:
        config.pretrain_path = args.pretrain
    if args.device:
        config.device = args.device

    trainer = Trainer(config)
    trainer.train()

if __name__ == "__main__":
    main()
