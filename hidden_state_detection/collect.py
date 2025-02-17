import json
import torch
import os
from tqdm import tqdm
from torch import nn
from transformers import AutoTokenizer
import pandas as pd
import random
random.seed(0)

def read_json(path):
    qa_data = []
    f = open(path, 'r', encoding='utf-8')
    for line in f.readlines():
        qa_data.append(json.loads(line))
    return qa_data

def write_jsonl(data, path):
    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    print(f'write jsonl to: {path}')
    f.close()

def arrange_hidden_states_for_single_layer(dir, total_layer):
    """
    得到每一层对应的hidden states和labels(labels对所有层相同)
    """
    paths = sorted([f for f in os.listdir(dir) if ".jsonl" in f])
    labels = [[] for _ in range(total_layer)]
    data = [[] for _ in range(total_layer)]
    for item in tqdm(paths):
        sub_data = read_json(os.path.join(dir, item))
        for idx in range(len(sub_data)):
            sample = sub_data[idx]
            for layer in range(total_layer):
                labels[layer].append(sample['has_answer'])
                data[layer].append(sample['hidden_states'][layer])

    for layer in range(total_layer):
        torch.save(torch.tensor(data[layer]), f'./data/layer{layer}/data.pt')
        torch.save(torch.tensor(labels[layer]), f'./data/layer{layer}/labels.pt')

def arrange_data_for_all_layers(dir, task):
    """
    得到所有样本对应的hidden states
    """
    paths = sorted([f for f in os.listdir(dir) if ".jsonl" in f])
    print(paths)
    labels = []
    data = []
    for item in tqdm(paths):
        sub_data = read_json(os.path.join(dir, item))
        for idx in range(len(sub_data)):
            sample = sub_data[idx]
            labels.append(sample['has_answer'])
            data.append(sample['hidden_states'])
    data = torch.tensor(data)
    labels = torch.tensor(labels)
    print(f'data shape: {data.shape}')
    print(f'label shape: {labels.shape}')
    torch.save(data, f'./data/{task}/all_layers/data.pt')
    torch.save(labels, f'./data/{task}/all_layers/labels.pt')

def arrange_probs(dir):
    labels = []
    data = []
    paths = sorted([f for f in os.listdir(dir) if ".jsonl" in f])
    for item in tqdm(paths):
        sub_data = read_json(os.path.join(dir, item))
        for idx in range(len(sub_data)):
            sample = sub_data[idx]
            labels.append(sample['has_answer'])
            data.append(sample['output_states'])
        torch.save(torch.tensor(data), f'./data/zero-shot-output/data.pt')
        torch.save(torch.tensor(labels), f'./data/zero-shot-output/labels.pt')
        
def save_all_data():
    for layer in range(33):
        if not os.path.exists(f'./data/layer{layer}'):
            os.mkdir(f'./data/layer{layer}')
    arrange_hidden_states_for_single_layer('./data/zero-shot-hidden', 33)

def dev_acc(dir):
    """
    得到所有数据的acc(所有任务), ref_label的平均概率, pred_label的平均概率, 返回每个任务的acc
    """
    res = []
    choices = {'A':0, 'B':1, 'C':2, 'D':3}
    avg_prob = []
    ref_prob = []
    paths = sorted([f for f in os.listdir(dir) if ".jsonl" in f])
    acc = [[] for _ in range(len(paths))]
    for task_id in tqdm(range(len(paths))):
        sub_data = read_json(os.path.join(dir, paths[task_id]))
        for sample in sub_data:
            acc[task_id].append(sample['has_answer'])
            res.append(sample['has_answer'])
            avg_prob.append(max(sample['Log_p']['token probs']))
            print(f"token probs: {sample['Log_p']['token probs']}")
            if sample['has_answer'] == 0:
                if len(sample['Log_p']['token probs']) == 4:
                    ref_prob.append(sample['Log_p']['token probs'][choices[sample['reference']]])
                else:
                    ref_prob.append(max(sample['Log_p']['token probs'][choices[sample['reference']]], sample['Log_p']['token probs'][choices[sample['reference']]+4]))
    acc = [sum(item) / len(item) for item in acc]
    print(f'dev count: {len(res)}')
    print(f'ref prob: {sum(ref_prob) / len(ref_prob)}')
    print(f'avg prob: {sum(avg_prob) / len(avg_prob)}')
    print(f'acc: {sum(res) / len(res)}')
    return acc

def split_all_data_to_task_data(all_data, all_labels, dir):
    """
    将所有数据按task拆分
    Input:
        - all_data: 所有数据
        - dir: 包含所有task的文件夹路径
    Return:
        - task_data:[[]], 按task拆分后的数据
        - paths: [], 所有task的名称
    """
    total_count = 0
    tasks_count = []
    # 统计各个task的样本数量
    paths = sorted([f for f in os.listdir(dir) if ".jsonl" in f])
    for item in tqdm(paths):
        data = read_json(os.path.join(dir, item))
        tasks_count.append(len(data))
        total_count += len(data)
    # 将all_data分配到各个task内
    task_data = [[] for _ in range(len(tasks_count))]
    task_label = [[] for _ in range(len(tasks_count))]
    task_id = 0
    count = tasks_count[0]
    for idx in range(len(all_data)):
        task_data[task_id].append(all_data[idx])
        task_label[task_id].append(all_labels[idx])
        count -= 1
        if count == 0 and task_id < len(tasks_count) - 1:
            task_id += 1
            count = tasks_count[task_id]
    return task_data, task_label, paths

def get_out_token_count(path):
    data = read_json(path)
    token_cnt = []
    for item in data:
        token_cnt.append(len(item['Log_p']['tokens']))
    print(f'avg token count: {sum(token_cnt)/len(token_cnt)}')

def get_res_for_different_seed(base_dir):
    """
    统计不同seed训练得到的结果
    Example:
        base_dir = './data/nq/llama3-8b-instruct/mid_layer/zero-shot-chat/mid_layer/res/'
        get_res_for_different_seed(base_dir)
    """
    total_score = []
    for mode in ['first', 'last', 'avg']:
        mode_score = [0.0, 0.0]
        for seed in ['0', '42', '100']:
            file_path = base_dir + 'sample_' + mode + '_seed' + seed + '.jsonl'
            if os.path.exists(file_path):
                pass
            else:
                file_path = base_dir + mode + '_seed' + seed + '.jsonl'
            data = read_json(file_path)
            for idx in [0, 1]:
                mode_score[idx] += list(data[idx].values())[0]
        mode_score = [round(item / 3, 4) for item in mode_score]
        print(f'{mode}-avg score: {mode_score}')
        total_score.append(mode_score)
    return [item[0] for item in total_score]

def get_conf_for_different_seed(conf_dir, label_path):
    labels = torch.load(label_path)
    conf_res = []
    overconf = []
    conserv = []
    for mode in ['first', 'last', 'avg']:
        mode_conf = []
        mode_overconf = []
        mode_conserv = []
        for seed in ['0', '42', '100']:
            file_path = conf_dir + 'pred_sample_' + mode + '_seed' + seed + '.jsonl'
            if os.path.exists(file_path):
                pass
            else:
                file_path = conf_dir + 'pred_' + mode + '_seed' + seed + '.jsonl'
            conf_data = read_json(file_path)[0]['test_pred']
            # print(conf_data)
            mode_conf.append(sum(conf_data) / len(conf_data))
            mode_overconf.append([conf_data[idx] > labels[idx] for idx in range(len(labels))])
            mode_overconf[-1] = sum(mode_overconf[-1]) / len(mode_overconf[-1]) # 每个seed的overcon
            mode_conserv.append([conf_data[idx] < labels[idx] for idx in range(len(labels))])
            mode_conserv[-1] = sum(mode_conserv[-1])/len(mode_conserv[-1]) # 每个seed的conserv
            
        conf_res.append(sum(mode_conf)/len(mode_conf)) # 3个seed平均conf
        overconf.append(sum(mode_overconf).item()/len(mode_overconf)) # 3个seed平均overconf
        conserv.append(sum(mode_conserv).item()/len(mode_conserv)) # 3个seed平均conserv
        conf_res = [round(item, 4) for item in conf_res]
        
        overconf = [round(item,4) for item in overconf]
        conserv = [round(item,4) for item in conserv]
        conf_right = [round(conf_res[idx] - overconf[idx],4) for idx in range(len(conf_res))]
        not_conf_rong = [round(1 - (conf_res[idx] + conserv[idx]),4) for idx in range(len(conf_res))]
    print(f'confidence: {conf_res}')
    print(f'conf_right: {conf_right}')
    print(f'overconf: {overconf}')
    print(f'not_conf_wrong: {not_conf_rong}')
    print(f'conserv: {conserv}')
    return conf_res, conf_right, overconf, not_conf_rong, conserv
                
def compute_acc(path):
    data = read_json(path)
    res = []
    for item in data:
        res.append(item['has_answer'])
    print(f'count: {len(res)}')
    print(f'acc: {sum(res)/len(res)}')

def different_knowledge_level():
    qa_data = read_json('./data/nq/llama3-8b-instruct/mid_layer/zero-shot-chat/nq_test_llama8b_tokens_mid_layer.jsonl')
    mc_rand_data = read_json('./data/nq-mc/llama3-8b-instruct/mid_layer/zero-shot-gene/nq-test-gene-choice.jsonl')
    assert len(qa_data) == len(mc_rand_data)
    right2wrong = []
    wrong2right = []
    for idx in range(len(qa_data)):
        if qa_data[idx]['has_answer'] == 1 and mc_rand_data[idx]['has_answer'] == 0:
            right2wrong.append(1)
        if qa_data[idx]['has_answer'] == 0 and mc_rand_data[idx]['has_answer'] == 1:
            wrong2right.append(1)
        print(qa_data[idx]['Res'])
    print(f'right->wrong: {round(len(right2wrong)/len(qa_data), 4)}')
    print(f'wrong->right: {round(len(wrong2right)/len(qa_data), 4)}')

def compute_acc_for_mc_task(ref_path, gene_path):
    """
    ref_path:数据集的path
    gene_path:跑出来结果的path
    """
    choices_idx = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
    ref_data = pd.read_csv(ref_path, header=None).to_numpy()
    gene_data = read_json(gene_path)
    acc = []
    assert len(ref_data) == len(gene_data)
    for idx in range(len(ref_data)):
        if gene_data[idx]['has_answer'] == 1 and ref_data[idx][choices_idx[ref_data[idx][-1]] + 1] != "None of the others":
            acc.append(1)
        else:
            acc.append(0)
    print(f'count: {len(acc)}')
    print(f'acc: {sum(acc)/len(acc)}')

def write_to_csv():
    total_score = []

    for model in ['llama2-chat-7b', 'llama3-8b-instruct', 'qwen2', 'llama2-chat-13b']:
        for chat_mode in ['zero-shot-3-random-1-gt-choose-gt', 'zero-shot-3-random-1-none-choose-none', 'zero-shot-3-gene-1-gt-choose-gt-gpt4o', 'zero-shot-3-gene-1-none-choose-none-gpt4o']:
            temp_score = []
            for dataset in ['nq-mc', 'hq-mc']:
                dir = f'../share/res/{dataset}/{model}/mid_layer/{chat_mode}/mid_layer/res/'
                temp_score += [0]
                temp_score += get_res_for_different_seed(dir)
            total_score.append(temp_score)
    # 将列表转换为DataFrame
    df = pd.DataFrame(total_score)

    # 保存为Excel文件
    df.to_excel('output.xlsx', index=False, header=False)

def get_test_label_for_mmlu(label_path, test_idx_path):
    labels = torch.load(label_path)
    test_idx = read_json(test_idx_path)
    test_labels = labels[test_idx]
    torch.save(test_labels, label_path.replace('label', 'test_label')) 

def get_conf_info_for_chat_and_cot():
    model = 'llama2-chat-7b'
    total_res = []
    acc = [0.2612, 0.1993, 0.422, 0.3643, 0.2955, 0.4551] # llama2
    # acc = [0.2753, 0.2163, 0.6249, 0.4435, 0.3679, 0.6377]
    # acc = [0.2731, 0.2496, 0.6872, 0.3776, 0.3334, 0.6863]
    # acc = [0.3227, 0.2369, 0.5058, 0.4199, 0.331, 0.5118]
    cnt = 0
    for chat_mode in ['zero-shot-chat', 'zero-shot-cot']:
        for dataset in ['nq', 'hq','mmlu']:
            conf_dir = f'../share/res/{dataset}/{model}/mid_layer/{chat_mode}/mid_layer/sample_res/'
            label_path = f'../share/res/{dataset}/{model}/mid_layer/{chat_mode}/mid_layer/test_labels.pt'
            conf, conf_right, overcon, not_con_wrong, conserv = get_conf_for_different_seed(conf_dir, label_path)
            for idx in range(len(conf)):
                total_res.append([conf[idx], round(conf_right[idx]/acc[cnt],4), overcon[idx], round(not_con_wrong[idx]/(1-acc[cnt]), 4), conserv[idx]])
            total_res.append([])
            cnt += 1
    print(total_res)
    df = pd.DataFrame(total_res)
    df.to_excel('output.xlsx', index=False, header=False)

if __name__ == '__main__':
    model = 'llama2-chat-7b'
    model_tail = {
        'llama2-chat-7b': 'llama7b',
        'llama3-8b-instruct': 'llama8b',
        'qwen2': 'qwen2',
        'llama2-chat-13b': 'llama13b'
    }
    # chat_mode = 'zero-shot-cot'
    # dataset = 'mmlu'
    # get_res_for_different_seed(f'../share/res/{dataset}-mc/{model}/mid_layer/{chat_mode}/mid_layer/res/')
    # compute_acc(f'../share/res/{dataset}-mc/{model}/mid_layer/{chat_mode}/{dataset}-test-gene-none.jsonl')

    # sample
    # for dataset in ['nq', 'hq']:
    #     for chat_mode in ['zero-shot-none']:
    #         train_sample_path = f'../share/res/{dataset}-mc/qwen2/mid_layer/{chat_mode}/{dataset}-train-none-choice.jsonl'
    #         sample_training_data_for_random_mc(train_sample_path, 1)

    # ref_path = '../share/datasets/hq-mc/test/hq-test-gene-choice-without-gt-4_test.csv'
    # gene_path= '../share/res/hq-mc/llama3-8b-instruct/mid_layer/zero-shot-wo-gt-4/hq-test-gene-choice-without-gt-4.jsonl'
    # compute_acc_for_mc_task(ref_path, gene_path)
    
    # conf, overconf, conserv
    total_res = []
    for dataset in ['nq', 'hq']:
        for cnt in [2, 4, 6, 8]:
            label_path = f'../share/res/{dataset}-mc/{model}/mid_layer/zero-shot-wo-gt-{cnt}-none-false-freeform-false-{model_tail[model]}/mid_layer/test_labels.pt'
            labels = torch.load(label_path)
            acc = (sum(labels)/len(labels)).item()
            print(f'acc for {model}-{dataset}-{cnt}: {acc}')
            conf_dir = f'../share/res/{dataset}-mc/{model}/mid_layer/zero-shot-wo-gt-{cnt}-none-false-freeform-false-{model_tail[model]}/mid_layer/sample_res/'
            conf, conf_right, overcon, not_con_wrong, conserv = get_conf_for_different_seed(conf_dir, label_path)
            for idx in range(len(conf)):
                total_res.append([conf[idx], 0, round(conf_right[idx]/acc,4), round(not_con_wrong[idx]/(1-acc), 4), overcon[idx], conserv[idx]])
            total_res.append([])
            cnt += 1
    print(total_res)
    df = pd.DataFrame(total_res)
    df.to_excel('output.xlsx', index=False, header=False)

    # 保存为Excel文件
    # total_res = []

    # for dataset in ['nq', 'hq','mmlu']:
    #     data_first = []
    #     data_last = []
    #     data_avg = []
    #     for model in ['llama2-chat-7b', 'llama3-8b-instruct', 'qwen2', 'llama2-chat-13b']:
    #             for chat_mode in ['zero-shot-chat', 'zero-shot-cot']:
    #                 align_dir = f'../share/res/{dataset}/{model}/mid_layer/{chat_mode}/mid_layer/sample_res/'
    #                 first_align, last_align, avg_align = get_res_for_different_seed(align_dir)
    #                 data_first.append(first_align)
    #                 data_last.append(last_align)
    #                 data_avg.append(avg_align)
    #     total_res.append(data_first)
    #     total_res.append(data_last)
    #     total_res.append(data_avg)

    # print(total_res)
    # df = pd.DataFrame(total_res)
    # df.to_excel('output.xlsx', index=False, header=False)



    


        






