import argparse
from utils.get_dataset import *
from sklearn.model_selection import train_test_split
from train import train_and_eval
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # prohibit parallelism
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def get_args():
    parser = argparse.ArgumentParser(description='News Binary Classification')

    # DataLoader
    parser.add_argument('--data-path', type=str, default='data_process/data.csv', help='path for the dataset')
    parser.add_argument('--batch-size', type=int, default=64, help='choose a batch size')
    parser.add_argument('--split-ratio', type=float, default=0.2, help='take a part of the whole dataset out')

    # Training Setting
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num-epochs', type=int, default=10, help='training epochs')
    parser.add_argument('--num-classes', type=int, default=2, help='output')

    # Optimizer
    # parser.add_argument('--optim', type=object, default='AdamW', help='optimizer for training')
    parser.add_argument('--lr', type=float, default='2e-4', help='learning rate for optimizer')

    # Models
    parser.add_argument('--model-name', type=str, default='TextCNN', help='choose a models')

    # Model hyperparams
    parser.add_argument('--num-filters', type=int, default=100, help='number of filters')
    parser.add_argument('--filter-sizes', type=list, default=[2, 3, 4], help='kinds of filter size')
    parser.add_argument('--embedding-size', type=int, default=128)
    parser.add_argument('--sequence-length', type=int, default=1000)
    parser.add_argument('--vocab-size', type=int, default=5000)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()

    # read data and buidl vocab
    df, word_index = load_tokenize_and_build_vocab(args)

    # split dataset into train and valid
    train_texts, valid_texts, train_labels, valid_labels = train_test_split(df['text'], df['label'],
                                                                            test_size=args.split_ratio, random_state=args.seed)
    # reset index
    train_texts, train_labels = train_texts.reset_index(drop=True), train_labels.reset_index(drop=True)
    valid_texts, valid_labels = valid_texts.reset_index(drop=True), valid_labels.reset_index(drop=True)

    # create dataset
    train_dataset = NewsDataset(args, train_texts, word_index, train_labels)
    valid_dataset = NewsDataset(args, valid_texts, word_index, valid_labels)

    # get dataloader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

    # train and eval
    train_and_eval(args, train_loader, valid_loader)

