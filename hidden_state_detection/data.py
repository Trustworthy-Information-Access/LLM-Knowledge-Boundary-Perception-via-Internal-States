import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from collect import read_json, write_jsonl
import json
import os
import random

class HiddenData:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def split_data_for_generation(data_path, label_path, need_layers):
    data = {
        'train': [],
        'dev': [],
        'test': []
    }
    labels = {
        'train': [],
        'dev': [],
        'test': []
    }
    for mode in ['train', 'dev', 'test']:  
        if mode != 'train':
            temp_data_path = data_path.replace('train', mode).replace('sample_', '')
            temp_label_path = label_path.replace('train', mode).replace('sample_', '')
        else:
            temp_data_path = data_path
            temp_label_path = label_path
        temp_data = torch.load(temp_data_path)
        data[mode] = torch.load(temp_data_path)[:, need_layers, :] if len(temp_data.shape) == 3 else torch.load(temp_data_path)
        temp_labels = torch.load(temp_label_path) # true/false
        labels[mode] = torch.zeros((len(temp_labels), 2))
        for idx in range(len(temp_labels)):
            labels[mode][idx][int(temp_labels[idx])] = 1
    train_data = data['train']
    train_labels = labels['train']
    dev_data = data['dev']
    dev_labels = labels['dev']
    test_data = data['test']
    test_labels = labels['test']
    print(f'train data: {train_data.shape}')
    print(f'train labels: {train_labels.shape}')
    print(f'dev data: {dev_data.shape}')
    print(f'test data: {test_data.shape}')
    return train_data, train_labels, dev_data, dev_labels, test_data, test_labels

def split_data_for_mmlu(data_path, label_path, need_layers, mmlu_train_idx):
    origin_data = torch.load(data_path)
    all_data = torch.load(data_path)[:, need_layers, :] if len(origin_data.shape) == 3 else torch.load(data_path)
    labels = torch.load(label_path) # true/false
    new_labels = torch.zeros((len(labels), 2))
    for idx in range(len(labels)):
        new_labels[idx][int(labels[idx])] = 1
    train_sample_idx = read_json('mmlu_train.jsonl')
    dev_sample_idx = read_json('mmlu_dev.jsonl')
    test_sample_idx = read_json('mmlu_test.jsonl')

    train_data = all_data[train_sample_idx]
    train_label = new_labels[train_sample_idx]
    dev_data = all_data[dev_sample_idx]
    dev_label = new_labels[dev_sample_idx]
    test_data = all_data[test_sample_idx]
    test_label = new_labels[test_sample_idx]

    print(f'train data: {train_data.shape}')
    print(f'train labels: {train_label.shape}')
    print(f'dev data: {dev_data.shape}')
    print(f'test data: {test_data.shape}')
    return train_data, train_label, dev_data, dev_label, test_data, test_label

def prepare_mode_data(path, hidden_modes, mode_hidden_states={}, labels=[]):
    """
    得到一个文件内所有数据的所有mode的hidden state,以及labels
    """
    data = read_json(path)
    if mode_hidden_states == {}:
        for mode in hidden_modes:
            mode_hidden_states[mode] = []

    for item in data:
        labels.append(item['has_answer'])
        for mode in hidden_modes:
            hidden_state = item['hidden_states'][mode]
            if len(hidden_state) != 0:
                mode_hidden_states[mode].append(hidden_state)
    return mode_hidden_states, labels

def prepare_mode_data_for_dir(dir, mode, hidden_modes):
    """
    得到mmlu的所有mode得到的hidden states
    为一个文件夹中的所有文件共同准备一个data.pt和label.pt
    """
    paths = [item for item in os.listdir(dir) if '.jsonl' in item]
    hidden_states = {}
    labels = []
    for path in paths:
        file_path = dir + path
        hidden_states, labels = prepare_mode_data(file_path, hidden_modes, hidden_states, labels)
    print(f'count: {len(labels)}')

    if not os.path.exists(dir + mode + '_layer/'):
        os.mkdir(dir + mode + '_layer/')
    for k, v in hidden_states.items():
        print(f'{k}: {len(v)}')
        if len(v) != 0:
            out_path = dir + mode + '_layer/' + k + '.pt'
            torch.save(torch.tensor(v), out_path)
    out_label = dir + mode + '_layer/labels.pt'
    torch.save(torch.tensor(labels), out_label)    

def prepare_mode_data_for_nq(dir, mode, hidden_modes):
    """
    为一个文件夹下所有文件都提取一个data.pt和label.pt
    """
    paths = [item for item in os.listdir(dir) if '.jsonl' in item and 'accuracy' not in item]
    for path in paths:
        file_path = dir + path
        hidden_states = {}
        labels = []
        hidden_states, labels = prepare_mode_data(file_path, hidden_modes, hidden_states, labels)
        print(f'count: {len(labels)}')
        if not os.path.exists(dir + mode + '_layer/'):
            os.mkdir(dir + mode + '_layer/')
        for k, v in hidden_states.items():
            print(f'{k}: {len(v)}')
            if len(v) != 0:
                out_path = dir + mode + '_layer/' + k + '_' +  path.replace('-', '_').split('_')[1] + '.pt'
                torch.save(torch.tensor(v), out_path)
        out_label = dir + mode + '_layer/'+ path.replace('-', '_').split('_')[1] + '_labels.pt'
        torch.save(torch.tensor(labels), out_label) 

def prepare_sample_train_data(train_path):
    """
    prepare .pt training data for sampled data
    """
    dir = '/'.join(train_path.split('/')[:-1]) + '/'
    mode = 'mid'
    hidden_states, labels = prepare_mode_data(train_path, ['first', 'avg', 'last'], {}, [])
    for k, v in hidden_states.items():
        print(f'{k}: {len(v)}')
        if len(v) != 0:
            out_path = dir + mode + '_layer/sample_' + k + '_train.pt'
            torch.save(torch.tensor(v), out_path)
    out_label = dir + mode + '_layer/sample_train_labels.pt'
    torch.save(torch.tensor(labels), out_label)   

def sample_training_data(data_path, acc=1, sample_cnt=1000):
    """
    sample 1000 right and 1000 wrong data for training
    """
    less_sample_list = []
    data = read_json(data_path)
    for idx in range(len(data)):
        if data[idx]['has_answer'] == acc:
            less_sample_list.append(idx)
    sample_cnt = len(less_sample_list) if len(less_sample_list) < sample_cnt else sample_cnt
    
    remain_idx = [item for item in range(len(data)) if item not in less_sample_list]
    total_idx = random.sample(less_sample_list, sample_cnt) + random.sample(remain_idx, sample_cnt)

    new_data = [data[idx] for idx in range(len(data)) if idx in total_idx]
    out_path = '/'.join(data_path.split('/')[:-1]) + '/' + data_path.split('/')[-1].replace('.jsonl', '-sample.jsonl')
    write_jsonl(new_data, out_path)

def sample_training_data_for_mmlu(train_idx_path, data_dir, sample_cnt=1000):
    paths = sorted([f for f in os.listdir(data_dir) if ".jsonl" in f and 'accuracy' not in f])
    all_data = []
    for item in paths:
        task_data = read_json(f'{data_dir}/{item}')
        for t_data in task_data:
            all_data.append(t_data)
    full_train_idx = read_json(train_idx_path)
    full_train_data = [all_data[idx] for idx in full_train_idx]
    wrong_train_idx = [idx for idx in range(len(full_train_data)) if full_train_data[idx]['has_answer'] == 0]
    right_train_idx = [idx for idx in range(len(full_train_data)) if full_train_data[idx]['has_answer'] == 1]
    sample_train_idx = random.sample(wrong_train_idx, sample_cnt) + random.sample(right_train_idx, sample_cnt)
    out_path = f'{data_dir}/mid_layer/sample_train_mmlu.jsonl'
    print(len(sample_train_idx))
    write_jsonl(sample_train_idx, out_path)

"""
Usage Example
"""

if __name__ == "__main__":
    model_tail = {
        'llama2-chat-7b': 'llama7b',
        'llama2-chat-13b': 'llama13b',
        'llama3-8b-instruct': 'llama8b',
        'qwen2': 'qwen7b'
    }
    for dataset in ['nq', 'hq', 'mmlu']:
        for chat_mode in ['zero-shot-chat', 'zero-shot-cot']:
            for model in ['llama2-chat-7b', 'llama3-8b-instruct', 'qwen2', 'llama2-chat-13b']:
                dir = f'../share/res/{dataset}/{model}/mid_layer/{chat_mode}/'
                hidden_mode = ['first', 'last', 'avg'] 
                # prepare_mode_data_for_nq(dir, 'mid', hidden_mode)
                # prepare_mode_data_for_dir(dir, 'mid', hidden_mode)
                # 提升用的文件的路径
                # file_name = chat_mode.replace('zero-shot-', '').replace('wo','without').replace('false','False').replace('true','True').replace('qwen2', 'qwen7b')
                # train_sample_path = f'../share/res/{dataset}-mc/{model}/mid_layer/{chat_mode}/{dataset}-train-gene-choice-{file_name}.jsonl'
                # 四种难度问题的路径
                # train_sample_path = f'../share/res/{dataset}-mc/{model}/mid_layer/{chat_mode}/{dataset}-train-random-none-choice.jsonl'
                # sample_training_data(train_sample_path, 0, 1000)
                # prepare_sample_train_data(train_sample_path.replace('.jsonl', '-sample.jsonl'))
                # 采样cot
                # train_sample_path = f'../share/res/{dataset}/{model}/mid_layer/{chat_mode}/{dataset}_train_{model_tail[model]}_tokens_cot_mid_layer.jsonl'
                # sample_training_data(train_sample_path, 0, 1000)
                # prepare_sample_train_data(train_sample_path.replace('.jsonl', '-sample.jsonl'))
                train_idx_path = './mmlu_train.jsonl'
                data_dir = f'../share/res/{dataset}/{model}/mid_layer/{chat_mode}'
                sample_training_data_for_mmlu(train_idx_path, data_dir, 1000)


    # train_sample_path='../share/res/hq/llama2-chat-13b/mid_layer/zero-shot-chat/hq_train_llama13b_tokens_mid_layer.jsonl'
    # sample_training_data(train_sample_path, 1)
    # prepare_sample_train_data(train_sample_path.replace('.jsonl', '-sample.jsonl'))