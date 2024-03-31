import argparse
from Dataloader import DataLoader
from Trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--path', type=str, default='./suv_data.csv')
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = args.path

    data = DataLoader(data_path)
    trainer = Trainer(data)
    trainer.train()


if __name__ == '__main__':
    main()