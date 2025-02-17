import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torch.optim import Adam
import os
import json
import argparse
from sklearn.model_selection import train_test_split
import re
import math
from sklearn.preprocessing import StandardScaler, MinMaxScaler

dev_score_list = []
dev_epoch_list = []
ood_score_list = []
ood_epoch_list = []
dev_pred_list = []
ood_pred_list = []

def read_json(path):
    qa_data = []
    f = open(path, 'r', encoding='utf-8')
    for line in f.readlines():
        qa_data.append(json.loads(line))
    return qa_data

class MLPNet(nn.Module):
    def __init__(self, dropout, in_dim=4096, out_dim=2):
        super(MLPNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, 64), # 4096 * 512
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(32, out_dim),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        x = self.layers(x)
        return x

def remove_punctuation_edges(s, name='movies'):
    """
    basketball数据集可能输出 "城市, 国家", 因此需要用逗号split
    """
    s = s.replace('\n', '')
    s = s.split('(')[0].strip()
    if name in ['basketball']:
        s = s.split(',')[0].strip()
    else:
        if len(s) <= 20:
            s = s.split(',')[0].strip()
    s = re.sub(r'^[^\w]+|[^\w]+$', '', s)
    s = s.strip()
    return s

class Postprocessor:
    def __init__(self, popularity_data, model_res, model) -> None:
        self.popularity_data = popularity_data
        self.model_res = model_res
        full_entities_dict = {}
        for d in self.popularity_data:
            full_entities_dict.update(d)
        self.full_entities_dict = full_entities_dict
        self.model = model

    def prepare_data(self, seed, test_ratio, dataset, type):
        all_conf = []
        all_acc = []
        all_question_popularity = []
        all_gene_popularity = []
        no_acc = []
        no_align = []
        no_conf = []

        for item in self.model_res:
            if item['Res'] == "" or item['Res'] == None:
                continue
            if item['popularity'] == "No":
                continue

            if 'gpt' in self.model:
                probs = [math.exp(t) for t in item['Log_p']['token_logprobs']]
                temp_conf = sum(probs)/len(probs)
                # temp_conf = sum(item['Log_p']['token_logprobs'])/len(item['Log_p']['token_logprobs'])
            else:
                temp_conf = sum(item['Log_p']['token_probs'])/len(item['Log_p']['token_probs'])

            gene_entity = remove_punctuation_edges(item['Res'], dataset)
            if self.full_entities_dict[gene_entity]['popularity'] == "No":
                no_acc.append(item['has_answer'])
                no_align.append(item['has_answer'] == 0)
                no_conf.append([temp_conf])
                continue

            gene_pop = self.full_entities_dict[gene_entity]['popularity']
            all_question_popularity.append(item['popularity'])
            all_conf.append(temp_conf)
            all_gene_popularity.append(gene_pop)
            all_acc.append(item['has_answer'])

        self.no_align = no_align

        return self.split_data(all_conf, all_gene_popularity, all_question_popularity, all_acc, type, seed, test_ratio)
        
    def split_data(self, confidence, popularity, question_pop, acc, type, seed=42, test_ratio=0.5):
        # 训练神经网络一定要特征归一化, 特征不一样scale影响巨大, 掉10个点
        scaler = MinMaxScaler()
        popularity = scaler.fit_transform([[item] for item in popularity])
        question_pop = scaler.fit_transform([[item] for item in question_pop])
        if type == 'conf_pop':
            x = [[confidence[i], popularity[i]] for i in range(len(confidence))]
        elif type == 'conf_question':
            x = [[confidence[i], question_pop[i]] for i in range(len(confidence))]
        else:
            x = [[confidence[i], popularity[i], question_pop[i]] for i in range(len(confidence))]

        y = [[0, 1] if t == 1 else [1, 0] for t in acc]
        assert len(acc) == len(confidence)


        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_ratio, random_state=seed)
        print(X_train[:10])
        print(sum([item[1] for item in y_train])/len(y_train))
        return torch.tensor(X_train).to(torch.float32), torch.tensor(X_test).to(torch.float32), torch.tensor(y_train).to(torch.float32), torch.tensor(y_test).to(torch.float32)
    
    def compute_alignment(self, best_test_pred, test_pred):
        best_test_align = best_test_pred + self.no_align
        test_align = test_pred + self.no_align

        return sum(best_test_align)/len(best_test_align), sum(test_align)/len(test_align)

class Generator:
    def __init__(self, train_data, test_data, batch_size, model):
        self.train_data = train_data
        self.test_data = test_data
        self.train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
        self.test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)
        self.model = model

    def finetune(self, epochs=100, mode='cnn', lr_rate=5e-5):
        device = torch.device('cuda')
        self.model = self.model.to(device)
        optimizer = Adam(self.model.parameters(), lr=lr_rate)
        criterion = torch.nn.CrossEntropyLoss()
        self.model.train()
        train_acc_list = []
        test_acc_list = []
        test_align_pred = []
        avg_test_loss, test_pred, _ = self.evaluate('test', mode, device)
        print('Begin Average test acc ' + str(avg_test_loss))

        for epoch in range(0, epochs):
            print('Epoch ' + str(epoch) + ' start')
            total_train_loss = 0.0
            acc = 0.0
            self.model.train()

            for step, batch in enumerate(self.train_loader):
                inputs = batch[0].to(device) # (batch_size, dim)
                target = batch[1].to(device)
                outputs = self.model(inputs)
                acc += sum(torch.argmax(outputs, axis=1) == torch.argmax(target, axis=1))
                loss = criterion(outputs, target)
                total_train_loss += loss
                loss.backward()
                optimizer.step()
                self.model.zero_grad()
            
            avg_train_loss = total_train_loss / len(self.train_loader)
            acc = acc / len(self.train_data)
            # print('Epoch ' + str(epoch + 1) + ', Average Train Loss ' + str(avg_train_loss))
            print('Epoch ' + str(epoch) + ', Average Train acc ' + str(acc.item()))
            # torch.save(self.model, model_save_path  + '_' + str(epoch + 1) + '_epoch')
            # eval
            avg_test_loss, test_pred, test_align = self.evaluate('test', mode, device)
            avg_train_loss, train_pred, _ = self.evaluate('train', mode, device)
            test_acc_list.append(avg_test_loss)
            test_align_pred.append(test_align)
            train_acc_list.append(avg_train_loss)
            print('Epoch ' + str(epoch) + ', Average dev acc ' + str(avg_test_loss.item()))
            print(f'pred right: {sum(test_pred)/len(test_pred)}')
        test_acc_list = torch.tensor(test_acc_list)
        train_acc_list = torch.tensor(train_acc_list)
        best_test_score, test_idx = torch.max(test_acc_list, dim=0)
        best_train_score, train_idx = torch.max(train_acc_list, dim=0)
        # print(f'test score: {test_score}, idx: {dev_idx}')
        # print(f'test max score: {best_test_score}, idx: {test_idx}')
        return best_test_score, test_idx, test_align_pred[test_idx], test_acc_list[train_idx], train_idx, test_align_pred[train_idx]

    def evaluate(self, test_mode, mode, device):
        self.model.eval()
        acc = 0
        data_loader = self.test_loader if test_mode == 'test' else self.train_loader
        data_set = self.test_data if test_mode == 'test' else self.train_data
        pred_list = []
        label_list = []
        align_list = []
        for batch in data_loader:
            inputs = batch[0].to(device)
            target = batch[1].to(device)
            target = target.to("cuda")
            target = torch.argmax(target, axis=1)
            pred = torch.argmax(self.model(inputs), axis=1)
            acc += sum(pred == target)
            pred_list += pred.tolist()
            label_list += target.tolist()
            align_list += [p == t for p, t in zip(pred, target)]
        return acc / len(data_set), pred_list, align_list

class PopData:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def run(args):
    mode = args.model
    dropout = args.dropout
    print('hyper-parameters-----------------------------------------------------------------------------')
    P = Postprocessor(read_json(args.pop_data), read_json(args.model_res), args.model_name)
    train_data, test_data, train_labels, test_labels = P.prepare_data(args.seed, 0.5, args.dataset, args.type)
    train_dataset = PopData(train_data, train_labels)
    test_dataset = PopData(test_data, test_labels)
    input_dim = train_data[0].shape[-1]
    net = MLPNet(dropout, input_dim)

    print(f'model: {net}')
    engine = Generator(train_dataset, test_dataset, args.batch_size, net)
    best_test_score, best_test_idx, best_test_pred, test_score, test_idx, test_pred = engine.finetune(epochs=args.epochs, mode=mode, lr_rate=args.lr_rate)

    best_test_all_align, test_all_align = P.compute_alignment(best_test_pred, test_pred)
    print(f'best test score: {best_test_score}, idx={best_test_idx}')
    print(f'best test all score: {best_test_all_align}, idx={best_test_idx}')
    print(f'test score: {test_score}, idx={test_idx}')
    print(f'test all score: {test_all_align}, idx={test_idx}')

    return best_test_score, best_test_all_align, test_score, test_all_align


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='mlp')
    parser.add_argument('--pop_data', type=str, default='xxx')
    parser.add_argument('--model_res', type=str, default='xxx')
    parser.add_argument('--model_name', type=str, default='xxx')
    parser.add_argument('--out_path', type=str, default='./score/')
    parser.add_argument('--dataset', type=str, default='movies')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr_rate', type=float, default=5e-5)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--which_gpu', type=str, default='0')
    parser.add_argument('--type', type=str, choices=['conf_pop', 'conf_question', 'conf_pop_question'], default='conf_pop_question')
    args = parser.parse_args()

    return args

def write_jsonl(data, path):
    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    print(f'write jsonl to: {path}')
    f.close()

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

    best_test_align, best_test_all_align, test_align, test_all_align = run(args)
    res = [{'best_test_score': round(best_test_align.item(), 4), 
            'best_test_all_score': round(best_test_all_align.item(), 4),
            'test_score': round(test_align.item(), 4),
            'test_all_score': round(test_all_align.item(), 4)}]

    # pred_res = [{'test_pred': test_pred}]
    if not os.path.exists(args.out_path + args.dataset):
        os.makedirs(args.out_path + args.dataset)
    write_jsonl(res, args.out_path + args.dataset + '/' + args.model_name + '_' + args.type + str(seed) + '.jsonl')
    # write_jsonl(pred_res, args.out_path + str(seed) + '_pred.jsonl')



