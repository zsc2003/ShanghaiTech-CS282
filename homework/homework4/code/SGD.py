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

    etas = [0.08, 0.04, 0.02, 0.01, 0.005, 0.0025]
    for eta in etas:
        trainer.train(eta)

    trainer.inference()
    trainer.print_info()

if __name__ == '__main__':
    main()