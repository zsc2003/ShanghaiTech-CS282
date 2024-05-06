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
    L1_lambda = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    L2_lambda = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

    trainer = Trainer(data, L1_lambda, L2_lambda)
    for L1_lam in L1_lambda:
        trainer.train('L1', L1_lam)
    for L2_lam in L2_lambda:
        trainer.train('L2', L2_lam)

    trainer.print_info()

if __name__ == '__main__':
    main()