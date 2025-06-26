import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import os
import json
from data import *
from models import *
from engine import Generator
import argparse
dev_score_list = []
dev_epoch_list = []
ood_score_list = []
ood_epoch_list = []
dev_pred_list = []
ood_pred_list = []

def run(args):
    hidden_size = args.hidden_size
    mode = args.model
    dropout = args.dropout
    num_layers=args.num_layers
    need_layers=args.need_layers
    print('hyper-parameters-----------------------------------------------------------------------------')
    print(f'hidden_size: {hidden_size}\nneed_layers: {need_layers}\ndropout: {dropout}')
    if 'nq' in args.data or 'hq' in args.data:
        train_data, train_labels, dev_data, dev_labels, test_data, test_labels = split_data_for_generation(args.data, args.label, need_layers)
    else:
        train_data, train_labels, dev_data, dev_labels, test_data, test_labels = split_data_for_mmlu(args.data, args.label, need_layers, args.mmlu_train_idx)
    train_dataset = HiddenData(train_data, train_labels)
    dev_dataset = HiddenData(dev_data, dev_labels)
    test_dataset = HiddenData(test_data, test_labels)
    input_dim = train_data[0].shape[-1]
    if mode == 'mlp' and len(train_data[0].shape) > 1:
        input_dim = train_data[0].shape[-1] * train_data[0].shape[-2] # dim * layers
    if mode == 'mlp':
        net = MLPNet(dropout, input_dim)
    else:
        raise ValueError("Specify a wrong model")
    print(f'model: {net}')
    engine = Generator(train_dataset, dev_dataset, test_dataset, args.batch_size, net)
    test_score, dev_idx, best_test_score, test_idx, test_pred, best_test_pred = engine.finetune(epochs=args.epochs, mode=mode, lr_rate=args.lr_rate)
    print(f'test score: {test_score}, idx={dev_idx}')
    print(f'best test score: {best_test_score}, idx={test_idx}')
    return test_score, best_test_score, test_pred


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='mlp')
    parser.add_argument('--data', type=str, default='./data/mmlu/llama2-chat-7b/zero-shot-chat/mid_layer/ans.pt')
    parser.add_argument('--label', type=str, default='./data/mmlu/llama2-chat-7b/zero-shot-chat/mid_layer/labels.pt')
    parser.add_argument('--out_path', type=str, default='./data/mmlu/llama2-chat-7b/zero-shot-chat/mid_layer/res.jsonl')
    parser.add_argument('--need_layers', type=int, default=-1)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--lr_rate', type=float, default=5e-5)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--which_gpu', type=str, default='1')
    parser.add_argument('--mmlu_train_idx', type=str, default='./mmlu_train.jsonl')
    args = parser.parse_args()

    return args
if __name__ == '__main__':
    args = get_args()
    seed=args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['CUDA_VISIBLE_DEVICES']=args.which_gpu
    print(args)

    score, best_score, test_pred = run(args)
    res = [{'test_score': round(score.item(), 4)}, {'best_test_score': round(best_score.item(), 4)}]
    pred_res = [{'test_pred': test_pred}]
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)
    hidden_mode = args.data.split('/')[-1].replace('.pt', '').replace('_train', '')
    write_jsonl(res, args.out_path + hidden_mode + '_seed' + str(seed) + '.jsonl')
    write_jsonl(pred_res, args.out_path + 'pred_' + hidden_mode + '_seed' + str(seed) + '.jsonl')


