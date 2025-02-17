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

def conver_hq_format(dir=''):
    data = json.loads(open('../hotpotqa/hotpot_train_v1.1.json').read())
    new_data = []
    for item in data:
        new_data.append({'question': item['question'], 'reference': item['answer'].split(', ')})
    print(len(new_data))
    write_jsonl(new_data, '../hotpotqa/hotpot_train.jsonl')

def write_2d_list_to_csv(filename, data):
    # 打开文件，使用 'w' 模式表示写入
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        
        # 逐行写入数据
        for row in data:
            writer.writerow(row)

def clean_data(ref, gene_str, model_name):
    if type(ref) != list:
        ref = [ref]
    if model_name not in ['llama7b', 'llama13b']:
        item_data = gene_str.replace('\n', '')
        # 使用正则表达式去除序号和多余空格
        cleaned_string = re.sub(r'\d+\.\s*', '', item_data).strip()
        # 以分号为分隔符进行分割并去除空白字符
        item_data = cleaned_string.split(':')
        # 生成格式可能是The answers are: , 也可能没有冒号
        item_data = item_data[0] if len(item_data) == 1 else ':'.join(item_data[1:])
        # 用完set顺序会变,需要控制以保证结果可复现
        # 选项不包含groun truth, 且不是拒绝回答
        gene_ans = list(dict.fromkeys([item.strip() for item in item_data.split(';') if len(item) > 0 and not has_answer(ref, item) and not deal_judge_new(item)]))
    else:
        # print(f'origin str: {gene_str}')
        item_data = gene_str.replace('\n\n', '\n')
        cleaned_string = re.sub(r'\d{1,2}\.\s*', '', item_data).strip()
        # print(f'clean_string: {cleaned_string}')
        if len(gene_str.split('\n')) >= 5:
            item_data = cleaned_string.split('\n')[1:]
        else:
            item_data = cleaned_string.split('\n')[-1].split(';')
        # print(f'split: {cleaned_string.split('\n')[1:]}')
        gene_ans = list(dict.fromkeys([item.split(';')[0].strip() for item in item_data if len(item) > 0 and not has_answer(ref, item) and not deal_judge_new(item)]))
        # print(f'ans: {gene_ans}')
    return gene_ans

def convert_qa_to_random_choices():
    dataset = 'hq'
    for mode in ['test', 'dev', 'train']:
        data = read_json(f'../{dataset}/{dataset}-{mode}.jsonl')
        csv_data = []
        for idx in range(len(data)):
            temp_answers = []
            idx_answer = {0:'A', 1:'B', 2:'C', 3:'D'}
            gt_choice='A'
            remain_idx = [item for item in range(len(data)) if item != idx]
            choose_answer = [data[item]['reference'][0] for item in random.sample(remain_idx, 3)]
            temp_answers.append(data[idx]['reference'][0])
            temp_answers += choose_answer
            random.shuffle(temp_answers)
            for ans_idx in range(len(temp_answers)):
                if temp_answers[ans_idx] == data[idx]['reference'][0]:
                    gt_choice = idx_answer[ans_idx]
            temp_answers.insert(0, data[idx]['question'])
            temp_answers.append(gt_choice)
            csv_data.append(temp_answers)
        write_2d_list_to_csv(f'../{dataset}/{dataset}-{mode}-random-choice.csv', csv_data)

def convert_random_gt_to_random_none():
    base_dir = '/Users/shiyuni/Documents/research/project/datasets'
    for dataset in ['nq', 'hq']:
        for mode in ['train', 'dev', 'test']:
            random_path = f'{base_dir}/{dataset}/{dataset}-{mode}-random-choice.csv'
            choice_idx = {'A':0, 'B':1, 'C':2, 'D':3}
            data = []  
            with open(random_path, mode='r') as file:
                csv_reader = csv.reader(file)
                # 遍历并打印每一行
                for row in csv_reader:
                    data.append(row)

            for idx in range(len(data)):
                data[idx][1 + choice_idx[data[idx][-1]]] = 'None of the others' 
            outdir = random_path.replace('random', 'random-none')
            write_2d_list_to_csv(outdir, data)



def convert_qa_to_generated_choices_with_gt(sample_cnt=3):
    random.seed(0)
    model_name='gpt4o'
    base_dir = '/Users/shiyuni/Documents/research/project/datasets'
    for dataset in ['nq', 'hq']:
        for mode in ['train', 'dev', 'test']:
            data = read_json(f'{base_dir}/{dataset}/{dataset}-{mode}.jsonl')
            gene_data = read_json(f'{base_dir}/{dataset}/10answers/{dataset}_{mode}_gpt4o_mini_10_answers.jsonl')
            assert len(data) == len(gene_data)
            csv_data = []
            cnt = 0
            for idx in range(len(data)):
                temp_answers = []
                temp_answers.append(data[idx]['reference'][0])
                idx_answer = {0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G', 7:'H'}
                gene_ans = clean_data(data[idx]['reference'], gene_data[idx]['Res'], 'gpt4o')
                need_cnt = 0
                if len(gene_ans) >= sample_cnt:
                    sample_data = random.sample(gene_ans, sample_cnt)
                else:
                    cnt += 1
                    sample_data = random.sample(gene_ans, len(gene_ans))
                    need_cnt = sample_cnt - len(gene_ans)
                    remain_idx = [item for item in range(len(data)) if item != idx]
                    # 生成的不够, 从其他问题答案中随机采样
                    remain_sample = [data[item]['reference'][0] for item in random.sample(remain_idx, need_cnt)]
                    sample_data += remain_sample
                sample_data.insert(0, data[idx]['reference'][0])
                random.shuffle(sample_data)
                for ans_idx in range(len(sample_data)):
                    if sample_data[ans_idx] == data[idx]['reference'][0]:
                        gt_choice = idx_answer[ans_idx]
                sample_data.insert(0, data[idx]['question'])
                sample_data.append(gt_choice)
                print(sample_data)
                csv_data.append(sample_data)
            print(cnt/len(gene_data))
            outdir = f'{base_dir}/{dataset}/{sample_cnt}-{model_name}-gene-1-gt'
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            
            write_2d_list_to_csv(f'{outdir}/{dataset}-{mode}-gene-choice-gpt4o_test.csv', csv_data)

def convert_qa_to_gene_none_data(): 
    """
    path = '/Users/shiyuni/Documents/research/project/datasets/hq/hq-test-gene-choice_test.csv'
    convert_qa_to_gene_none_data(path)
    """
    model_name='gpt4o'
    base_dir='/Users/shiyuni/Documents/research/project/datasets'
    chat_mode = f'3-{model_name}-gene-1-gt'
    for dataset in ['hq']:
        for data_mode in ['train', 'test', 'dev']:
            path = f'{base_dir}/{dataset}/{chat_mode}/{dataset}-{data_mode}-gene-choice-{model_name}_test.csv'
            choice_idx = {'A':0, 'B':1, 'C':2, 'D':3}
            data = []  
            with open(path, mode='r') as file:
                csv_reader = csv.reader(file)
                # 遍历并打印每一行
                for row in csv_reader:
                    data.append(row)

            for idx in range(len(data)):
                data[idx][1 + choice_idx[data[idx][-1]]] = 'None of the others' 
            outdir = f'{base_dir}/{dataset}/{chat_mode.replace('gt', 'none')}'
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            out_path = f'{base_dir}/{dataset}/{chat_mode.replace('gt', 'none')}/{dataset}-{data_mode}-none-choice-{model_name}_test.csv'
            write_2d_list_to_csv(out_path, data)

def convert_qa_to_generated_choices(ans_cnt=3, with_none=False, freeform=True, rank_acc=[]):
    """
    一共有两种setting, 两种setting相互独立
        - 第一种: 是否在选项中添加none选项
        - 第二种: 选项中是否添加free-form形式时生成的答案   
    -Example:
        convert_qa_to_generated_choices(8, False, False)
    """
    random.seed(0)
    model_name='llama8b'
    base_dir = '/Users/shiyuni/Documents/research/project/datasets'
    for mode in ['train', 'dev', 'test']:
        for dataset in ['nq', 'hq']:
            data = read_json(f'{base_dir}/{dataset}/{dataset}-{mode}.jsonl')
            origin_gene_data = read_json(f'{base_dir}/{dataset}/10answers/{dataset}_{mode}_{model_name}_10_answers.jsonl')
            if freeform == True:
                freeform_data = read_json(f'{base_dir}/{dataset}/{model_name}/{dataset}_{mode}_{model_name}.jsonl')
                assert len(freeform_data) == len(data)
            assert len(data) == len(origin_gene_data)
    
            csv_data = []
            cnt = 0
            acc = 0
            freeform_acc=0
            gene_data = [clean_data('nishiyu', item['Res'], model_name) for item in origin_gene_data]
            unique_choice_cnt = [len(item) for item in gene_data]
            print(sum(unique_choice_cnt)/len(unique_choice_cnt))
            for idx in range(len(data)):
                right_answer = ""
                idx_answer = {-1: 'none', 0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G', 7:'H', 8:'I', 9:'J'}
                if freeform == True:
                    freeform_ans = freeform_data[idx]['Res']
                    # 与freeform_ans一样的不能加进来
                    gene_ans = [item for item in gene_data[idx] if not has_answer([freeform_ans.replace('.', '').replace('\n', '')], item) and not has_answer([item], freeform_ans)]
                    freeform_acc += freeform_data[idx]['has_answer']
                else:
                    gene_ans = gene_data[idx]


                if freeform == True:
                    sample_cnt = ans_cnt - 1
                else:
                    sample_cnt = ans_cnt # 取n个选项
                need_cnt = 0
                gt_choice_idx = -1
                # 先凑齐需要的生成choice
                if len(gene_ans) >= sample_cnt:
                    sample_data = gene_ans[:sample_cnt]
                else:
                    cnt += 1
                    sample_data = gene_ans
                    need_cnt = sample_cnt - len(sample_data)
                    remain_idx = [item for item in range(len(data)) if item != idx and len(gene_data[item]) >= 1]
                    # 生成的不够, 从其他问题答案中随机采样
                    remain_ans = [gene_data[item][0] for item in random.sample(remain_idx, need_cnt)]
                    sample_data += remain_ans
                if freeform == True:
                    sample_data.append(freeform_ans)
                # 凑齐了answer选项, 进行下一步
                for item in sample_data:
                    if has_answer(data[idx]['reference'], item):
                        right_answer = item
                        acc += 1
                        break
                # 需要加一个none选项进去    
                if with_none == True:
                    if right_answer == "":
                        right_answer = 'None of the others'
                    sample_data.append('None of the others')

                random.shuffle(sample_data)
                for choice_idx in range(len(sample_data)):
                    if sample_data[choice_idx] == right_answer:
                        gt_choice_idx = choice_idx
                sample_data.append(idx_answer[gt_choice_idx])
                sample_data.insert(0, data[idx]['question'])
                csv_data.append(sample_data)
                # print(sample_data)
            print(f'less answer ratio: {cnt/len(gene_data)}')
            print(f'acc: {acc/len(data)}')
            print(f'freeform_acc:{freeform_acc/len(data)}')
            rank_acc.append(acc/len(data))
            # chat_mode = f'wo-gt-{ans_cnt}-none-{with_none}-freeform-{freeform}-{model_name}'
            # if not os.path.exists(f'{base_dir}/{dataset}/{chat_mode.lower()}'):
            #     os.makedirs(f'{base_dir}/{dataset}/{chat_mode.lower()}')
            # out_path = f'{base_dir}/{dataset}/{chat_mode.lower()}/{dataset}-{mode}-gene-choice-without-gt-{ans_cnt}-none-{with_none}-freeform-{freeform}-{model_name}_test.csv'
            # write_2d_list_to_csv(out_path, csv_data)


def convert_to_multi_round_data():
    """
    获得multi-round confidence extraction的数据, 构造数据为
    question:[
        1. question
        2. response
        3. Generate 10 possible answers
        4. response
    ]
    """
    model_name='qwen7b'
    base_dir = '/Users/shiyuni/Documents/research/project/datasets'
    for mode in ['train', 'dev', 'test']:
        for dataset in ['nq', 'hq']:
            qa_data = read_json(f'{base_dir}/{dataset}/{model_name}/{dataset}_{mode}_{model_name}.jsonl')
            more_ans_data = read_json(f'{base_dir}/{dataset}/10answers/{dataset}_{mode}_{model_name}_10_answers.jsonl')
            all_data = []
            assert len(qa_data) == len(more_ans_data)
            for idx in range(len(qa_data)):
                multi_round_data = []
                multi_round_data.append(qa_data[idx]['question'])
                multi_round_data.append(qa_data[idx]['Res'])
                multi_round_data.append('Generate 10 possible answers for the following question, each separated by a semicolon')
                multi_round_data.append(more_ans_data[idx]['Res'])
                print(multi_round_data)
                all_data.append({'question': multi_round_data, 'reference': qa_data[idx]['reference']})
                out_path = f'{base_dir}/{dataset}/multi_round/{dataset}_{mode}_{model_name}.jsonl'
            write_jsonl(all_data, out_path)

def replace_random_to_gene():
    base_dir = '/Users/shiyuni/Documents/research/project/datasets'
    for dataset in ['nq', 'hq']:
        for mode in ['train', 'dev', 'test']:
            random_path = f'{base_dir}/{dataset}/3-random-1-gt-choose-gt/{dataset}-{mode}-random-choice.csv'
            gene_path = f'{base_dir}/{dataset}/3-gpt4o-gene-1-gt/{dataset}-{mode}-gene-choice-gpt4o_test.csv'
            choice_idx = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4}
            rand_data = [] 
            gene_data = [] 
            with open(random_path, mode='r') as file:
                csv_reader = csv.reader(file)
                # 遍历并打印每一行
                for row in csv_reader:
                    rand_data.append(row)

            with open(gene_path, mode='r') as file:
                csv_reader = csv.reader(file)
                # 遍历并打印每一行
                for row in csv_reader:
                    gene_data.append(row)
            

            for idx in range(len(rand_data)):
                rand_not_gt_idx = [item for item in range(4) if item != choice_idx[rand_data[idx][-1]]]
                gene_not_gt_ans = [item for item in range(4) if item != choice_idx[gene_data[idx][-1]]]
                rand_data[idx][1 + rand_not_gt_idx[0]] = gene_data[idx][1 + gene_not_gt_ans[0]]
            out_dir = f'{base_dir}/{dataset}/2-random-1-gene-1-gt-choose-gt'
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            outpath = f'{out_dir}/{dataset}-{mode}-random-gene-choice.csv'
            write_2d_list_to_csv(outpath, rand_data)

def get_choics_cnt_and_acc():
    model_names = ['llama7b', 'llama8b', 'qwen7b', 'llama13b']
    base_dir = '/Users/shiyuni/Documents/research/project/datasets'
    res = []
    res_acc = []
    for dataset in ['nq', 'hq']:
        for model_name in model_names:
            all_choice_cnt = []
            top_k_acc = []
            for mode in ['train', 'dev', 'test']:
                
                data = read_json(f'{base_dir}/{dataset}/{dataset}-{mode}.jsonl')
                origin_gene_data = read_json(f'{base_dir}/{dataset}/10answers/{dataset}_{mode}_{model_name}_10_answers.jsonl')
                gene_data = [clean_data('nishiyu', item['Res'], model_name) for item in origin_gene_data]
                unique_choice_cnt = [len(item) for item in gene_data]
                all_choice_cnt.append(round(sum(unique_choice_cnt)/len(unique_choice_cnt), 2))
                print(all_choice_cnt)
                if mode in ['train', 'dev']:
                    continue
                for ans_cnt in [2, 4, 6, 8]:
                    temp_acc = 0

                    for idx in range(len(data)):
                        gene_ans = gene_data[idx]
                        sample_cnt = ans_cnt # 取n个选项
                        need_cnt = 0
                    # 先凑齐需要的生成choice
                        if len(gene_ans) >= sample_cnt:
                            sample_data = gene_ans[:sample_cnt]
                        else:
                            sample_data = gene_ans
                            need_cnt = sample_cnt - len(sample_data)
                            remain_idx = [item for item in range(len(data)) if item != idx and len(gene_data[item]) >= 1]
                            # 生成的不够, 从其他问题答案中随机采样
                            remain_ans = [gene_data[item][0] for item in random.sample(remain_idx, need_cnt)]
                            sample_data += remain_ans

                        for item in sample_data:
                            if has_answer(data[idx]['reference'], item):
                                temp_acc += 1
                                break
                    top_k_acc.append(round(temp_acc/len(data) * 100, 2))
            res.append(all_choice_cnt)
            res_acc.append(top_k_acc)
    print(res)
    print(res_acc)
    write_2d_list_to_csv('./unique_gene_ans_cnt.csv', res)
    write_2d_list_to_csv('./gene_ans_acc.csv', res_acc)
    

if __name__ == '__main__':
    res=[]
    for num in [2, 4, 6, 8]:
        convert_qa_to_generated_choices(num, False, False, res)
    print(res)
    # convert_qa_to_generated_choices_with_gt()
    # convert_random_gt_to_random_none()
    # replace_random_to_gene()
    # get_choics_cnt_and_acc()



