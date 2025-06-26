import json
import random
from utils import has_answer, deal_judge_new
import csv
import re
import os
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

def write_2d_list_to_csv(filename, data):
    # 打开文件，使用 'w' 模式表示写入
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        for row in data:
            writer.writerow(row)

def clean_data(ref, gene_str, model_name):
    if type(ref) != list:
        ref = [ref]
    if model_name not in ['llama7b', 'llama13b']:
        item_data = gene_str.replace('\n', '')
        cleaned_string = re.sub(r'\d+\.\s*', '', item_data).strip()
        item_data = cleaned_string.split(':')
        item_data = item_data[0] if len(item_data) == 1 else ':'.join(item_data[1:])
        gene_ans = list(dict.fromkeys([item.strip() for item in item_data.split(';') if len(item) > 0 and not has_answer(ref, item) and not deal_judge_new(item)]))
    else:
        item_data = gene_str.replace('\n\n', '\n')
        cleaned_string = re.sub(r'\d{1,2}\.\s*', '', item_data).strip()
        if len(gene_str.split('\n')) >= 5:
            item_data = cleaned_string.split('\n')[1:]
        else:
            item_data = cleaned_string.split('\n')[-1].split(';')
        gene_ans = list(dict.fromkeys([item.split(';')[0].strip() for item in item_data if len(item) > 0 and not has_answer(ref, item) and not deal_judge_new(item)]))
    return gene_ans

def convert_qa_to_generated_choices(model_name, base_dir, ans_cnt=3, rank_acc=[]):
    """
    convert model-generated string to multi-choice questions.
    """
    random.seed(0)
    for mode in ['train', 'dev', 'test']:
        for dataset in ['nq', 'hq']:
            data = read_json(f'{base_dir}/{dataset}/{dataset}-{mode}.jsonl')
            csv_data = []
            cnt = 0
            acc = 0
            origin_gene_data = read_json(data)
            gene_data = [clean_data('nishiyu', item['Res'], model_name) for item in origin_gene_data]
            unique_choice_cnt = [len(item) for item in gene_data]
            print(sum(unique_choice_cnt)/len(unique_choice_cnt)) # avg unique candidate answers
            for idx in range(len(data)):
                right_answer = ""
                idx_answer = {-1: 'none', 0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G', 7:'H', 8:'I', 9:'J'}
                gene_ans = gene_data[idx]
                sample_cnt = ans_cnt # count of answers need to choose
                need_cnt = 0
                gt_choice_idx = -1

                if len(gene_ans) >= sample_cnt:
                    sample_data = gene_ans[:sample_cnt]
                else:
                    cnt += 1
                    sample_data = gene_ans
                    need_cnt = sample_cnt - len(sample_data)
                    remain_idx = [item for item in range(len(data)) if item != idx and len(gene_data[item]) >= 1]
                    # if there is no sufficient generated answers, choose from answers corresponding to other questions
                    remain_ans = [gene_data[item][0] for item in random.sample(remain_idx, need_cnt)]
                    sample_data += remain_ans

                for item in sample_data:
                    # compute whether sample_data contains the correct answer.
                    if has_answer(data[idx]['reference'], item):
                        right_answer = item
                        acc += 1
                        break
                # shuffle
                random.shuffle(sample_data)
                for choice_idx in range(len(sample_data)):
                    if sample_data[choice_idx] == right_answer:
                        gt_choice_idx = choice_idx
                sample_data.append(idx_answer[gt_choice_idx])
                sample_data.insert(0, data[idx]['question'])
                csv_data.append(sample_data)
            print(f'less answer ratio: {cnt/len(gene_data)}')
            print(f'acc: {acc/len(data)}')
            rank_acc.append(acc/len(data))
            chat_mode = f'{model_name}-{ans_cnt}'
            if not os.path.exists(f'{base_dir}/{dataset}/{chat_mode.lower()}'):
                os.makedirs(f'{base_dir}/{dataset}/{chat_mode.lower()}')
            out_path = f'{base_dir}/{dataset}/{chat_mode.lower()}/multi-choice-questions-{mode}.csv'
            write_2d_list_to_csv(out_path, csv_data)
    

if __name__ == '__main__':
    base_dir="xxx"
    model_name='xxx'
    res=[]
    for num in [2, 4, 6, 8]:
        convert_qa_to_generated_choices(model_name, base_dir, num, res)
    print(res)




