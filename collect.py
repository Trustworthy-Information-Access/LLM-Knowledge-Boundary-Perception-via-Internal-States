from utils.utils import read_json, write_jsonl, has_answer
import os
import pandas as pd
import torch

def compute_one_file(path):
    data = read_json(path)
    entro_list = []
    acc_list = []
    for idx in range(len(data)):
        sample = data[idx]
        entro_list.append(sample['Log_p']['token_entropy'])
        acc_list.append(sample['has_answer'])
    return sum(acc_list)/len(acc_list), sum(entro_list)/len(entro_list)

def compute_all_files(dir):
    choice_idx = {'A':0, 'B':1, 'C':2, 'D':3}
    paths = sorted([f for f in os.listdir(dir) if ".jsonl" in f])
    entro_list, acc_list, token_prob, ref_prob = [], [], [], []
    for item in paths:
        data = read_json(os.path.join(dir, item))
        temp_acc = []
        for idx in range(len(data)):
            sample = data[idx]
            # if sample['has_answer'] != 1:
            #     continue
            temp_acc.append(sample['has_answer'])
            entro_list.append(sample['Log_p']['token_entropy'])
            acc_list.append(sample['has_answer'])
            token_prob.append(sample['Log_p']['token probs'][choice_idx[sample['Res']]])
            ref_prob.append(sample['Log_p']['token probs'][choice_idx[sample['reference']]])
        print(sum(temp_acc) / len(temp_acc))
    print(f'count: {len(acc_list)}')
    print(f'avg acc: {sum(acc_list)/len(acc_list)}')
    print(f'avg entropy: {sum(entro_list)/len(entro_list)}')
    print(f'avg token prob: {sum(token_prob) / len(token_prob)}')
    print(f'avg ref prob: {sum(ref_prob) / len(ref_prob)}')
    return acc_list, token_prob

def compute_ece(acc_list, prob_list):
    acc_bin = [[] for _ in range(10)]
    prob_bin = [[] for _ in range(10)]
    for idx in range(len(prob_list)):
        bin_idx = int(prob_list[idx] * 10)
        acc_bin[bin_idx].append(acc_list[idx])
        prob_bin[bin_idx].append(prob_list[idx])
    for idx in range(len(acc_bin)):
        if len(acc_bin[idx]) == 0:
            continue
        print(f'bin {idx}')
        print(f'count: {len(acc_bin[idx])}')
        print(f'avg acc: {sum(acc_bin[idx])/len(acc_bin[idx])}')
        print(f'avg prob: {sum(prob_bin[idx])/len(prob_bin[idx])}')

def get_align_for_verbalized_conf(acc_path, verb_conf):
    answer_data = read_json(acc_path)
    conf_data = read_json(verb_conf)
    align = []
    assert len(answer_data) == len(conf_data)
    for idx in range(len(answer_data)):
        if answer_data[idx]['has_answer'] != conf_data[idx]['has_answer']:
            align.append(1)
        else:
            align.append(0)
    print(f'count: {len(align)}')
    print(f'align: {sum(align)/len(align)}')

def get_align_for_verbalized_conf_for_dir(acc_dir, verb_dir):
    acc_list = []
    conf_list = []
    align = []
    acc_paths = sorted([f for f in os.listdir(acc_dir) if ".jsonl" in f])
    for item in acc_paths:
        acc_data = read_json(os.path.join(acc_dir, item))
        conf_data = read_json(os.path.join(verb_dir, item))
        for idx in range(len(acc_data)):
            acc_list.append(acc_data[idx]['has_answer'])
            conf_list.append(conf_data[idx]['has_answer'])
            if acc_list[-1] != conf_list[-1]:
                align.append(1)
            else:
                align.append(0)
    print(f'count: {len(acc_list)}')
    print(f'acc: {sum(acc_list)/len(acc_list)}')
    print(f'uncertain: {sum(conf_list)/len(conf_list)}')
    print(f'align: {sum(align)/len(align)}')

def consistency_for_different_fils(question_path_list, freeform_path, mc_ans_path_list):
    """
    是否都选择了freeform中生成的选项
    - data_path:构造的各种选择题文件
    - ans_conf_path_list:第一个元素是freeform文件的path,后面是构造成各种选择题的文件的path
    还没写完
    """
    # freeform的在第一个
    choice2answer = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4, 'F':5, 'G':6, 'H':7}
    question = []
    # 先提取freeform ans
    freeform_data = read_json(freeform_path)
    freeform_ans = [item['Res'] for item in freeform_data]
    # 提取每个选择题选择的答案
    # 1.提取问题
    for path in question_path_list:
        question.append(pd.read_csv(path, header=None).to_numpy())
    # 2.提取输出选项对应的答案

    total_choose_the_same = []
    for path_idx in range(len(mc_ans_path_list)):
        choose_the_same = []
        mc_res = read_json(mc_ans_path_list[path_idx])
        for idx in range(len(mc_res)):
            ans_idx = choice2answer[mc_res[idx]['Res']] + 1
            choose_ans = str(question[path_idx][idx][ans_idx])

            if has_answer([choose_ans], freeform_ans[idx]) or has_answer([freeform_ans[idx]], choose_ans):
                choose_the_same.append(1)

            else:
                choose_the_same.append(0)
        total_choose_the_same.append(choose_the_same)
        print(f'choose the same: {[sum(item)/len(item) for item in total_choose_the_same]}')
    return total_choose_the_same



def alignment_match_for_different_files(freeform_conf_path, ans_conf_path_list, conf_label_path):
    """
    1.聚合多个问题的confidence并返回二维列表,其中第一个元素是freeform下的conf_list
    2.分析不同难度选择题与freeform下conf的consistent程度
    Input
        - freeform_align_path: freeform下对测试集的conf_path
        - ans_conf_path_list: 多种难度的选择题形式下的conf_list组成的二维列表
        - conf_label_path: freefrom下测试集的真实conf_label_path
    Return
        - ans_conf: 所有形式conf的二维列表,第一个元素为freeform对应的列表
        - gt_conf: freeform下的测试集ground truth conf列表
    """
    freeform_conf = read_json(freeform_conf_path)[0]['test_pred']
    gt_conf = torch.load(conf_label_path).tolist()
    ans_conf = []

    for path in ans_conf_path_list:
        temp_conf = read_json(path)[0]['test_pred']
        ans_conf.append(temp_conf)
        # print(f'consis align: {sum(consis_align)/len(consis_align)}')
        # print(f'not consis align: {sum(not_consis_align)/len(not_consis_align)}')
    ans_conf.insert(0, freeform_conf)
    return ans_conf, gt_conf

def majority_vote(lists):
    """
    大多数投票算法
    """
    # 假设所有列表长度相同，获取第一个列表的长度
    list_length = len(lists[0])
    
    # 保存投票结果
    result = []
    align = []
    
    # 对每个位置进行大多数投票
    for i in range(list_length):
        # 统计当前位置上的 0 和 1 的数量
        vote_count = sum([lst[i] for lst in lists])
        
        # 如果超过一半是1，就将该位置的值设为1，否则设为0
        if vote_count > len(lists) / 2:
            result.append(1)
        else:
            result.append(0)
    return result

def rule_based_cooperate(conf_lists, gt_conf, consis_lists=[]):
    """
    基于规则的,在多个conf_list间相互配合
    Input:
        - conf_lists: 多个难度的问题下,对结果的confidence list
        - gt_conf: 测试集freeform真实accuracy
    """
    consis_list = []
    not_consis_list = []
    overcon = []
    conserv = []
    info = []
    safe = []
    real_wrong_in_not_conf = []
    for i in range(len(conf_lists[0])): # 第i个问题
        vote_num = sum([lst[i] for lst in conf_lists])
        if vote_num == len(conf_lists) or vote_num == 0: # freeform与mc的判断都一致
            consis_list.append(conf_lists[0][i] == gt_conf[i])
        else:
            # 这个规则不能加,加了效果会变差
            # if conf_lists[0][i] == 0 and vote_num >=4:
                # conf_lists[0][i] = 1
                # real_wrong_in_not_conf.append(gt_conf[i])
                
            # 加一个规则,若freeform认为没做对,但选择题做对了,且选择答案就是freeform里的答案,则freeform可能判断错误
            # if conf_lists[0][i] == 0:
            #     # 第t种方式的第i个问题
            #     # 选的答案和freeform一样,并且还认为做对了的
            #     consis_and_right = [consis_lists[t][i] and conf_lists[t+1][i] for t in range(len(consis_lists))]
                
                # if sum(consis_and_right) >= 4:
                #     real_wrong_in_not_conf.append(gt_conf[i])

            # 若freeform时判断能做对,但其余形式都做不对,则freeform判断可能错误
            if conf_lists[0][i] == 1 and vote_num <=1:
                conf_lists[0][i] = 0
            not_consis_list.append(conf_lists[0][i] == gt_conf[i])

        overcon.append(conf_lists[0][i] > gt_conf[i])
        conserv.append(conf_lists[0][i] < gt_conf[i])
        info.append(conf_lists[0][i] == 1 and gt_conf[i] == 1)
        safe.append(conf_lists[0][i] == 0 and gt_conf[i] == 0)
        
    # print(f'real_wrong_in_not_conf: {sum(real_wrong_in_not_conf)/len(real_wrong_in_not_conf)}')
    total_align = consis_list + not_consis_list
    return sum(total_align)/len(total_align), sum(conf_lists[0])/len(conf_lists[0]), sum(info)/sum(gt_conf), sum(safe)/(len(gt_conf)-sum(gt_conf)), sum(overcon)/len(overcon), sum(conserv)/len(conserv)

def rule_based_alignment_for_different_seed():
    """
    计算各个seed下,基于规则的聚合方法的alignmen平均值
    """
    # model = 'llama3-8b-instruct'
    dataset = 'nq'
    model_tail = {
        'llama2-chat-7b': 'llama7b',
        'llama3-8b-instruct': 'llama8b',
        'qwen2': 'qwen2',
        'llama2-chat-13b': 'llama13b'
    }
    save_res = []
    for model in ['llama2-chat-7b', 'llama3-8b-instruct', 'qwen2', 'llama2-chat-13b']:
        # freeform_ans = f'../share/res/{dataset}/{model}/mid_layer/zero-shot-chat/{dataset}_test_{model_tail[model]}_tokens_mid_layer.jsonl'
        conf_label_path = f'../share/res/{dataset}/{model}/mid_layer/zero-shot-chat/mid_layer/test_labels.pt'
        for mode in ['last']:
            total_align = []
            total_conf = []
            total_info = []
            total_safe = []
            total_overc = []
            total_conserv = []
            for seed in [0, 42, 100]:
                freeform_conf = f'../share/res/{dataset}/{model}/mid_layer/zero-shot-chat/mid_layer/sample_res/pred_sample_{mode}_seed{seed}.jsonl'
                ans_conf_path = []
                ans_path = []
                question_path = []
                for cnt in [2, 4, 6, 8]: # mc_ans path
                    ans_conf_path.append(f'../share/res/{dataset}-mc/{model}/mid_layer/zero-shot-wo-gt-{cnt}-none-false-freeform-false-{model_tail[model]}/mid_layer/sample_res/pred_sample_{mode}_seed{seed}.jsonl')
                    # question_name = f'wo-gt-{cnt}-none-false-freeform-false-{model_tail[model]}'.replace('wo','without').replace('false','False').replace('qwen2', 'qwen7b')
                    # ans_path.append(f'../share/res/{dataset}-mc/{model}/extract/{dataset}-test-{cnt}.jsonl')
                    # question_path.append(f'../share/datasets/{dataset}-mc/wo-gt-{cnt}-none-false-freeform-false-{model_tail[model]}/{dataset}-test-gene-choice-{question_name}_test.csv')
                # consis_list = consistency_for_different_fils(question_path, freeform_ans, ans_path)
                all_conf_list, gt_conf = alignment_match_for_different_files(freeform_conf, ans_conf_path, conf_label_path)
                seed_align, seed_conf, seed_info, seed_safe, seed_overc, seed_consev = rule_based_cooperate(all_conf_list, gt_conf, [])
                total_align.append(seed_align)
                total_conf.append(seed_conf)
                total_info.append(seed_info)
                total_safe.append(seed_safe)
                total_overc.append(seed_overc)
                total_conserv.append(seed_consev)
            mode_res = [sum(total_align)/len(total_align), sum(total_conf)/len(total_conf), sum(total_safe)/len(total_safe), sum(total_overc)/len(total_overc)]
            mode_res = [round(item, 4) for item in mode_res]
            save_res.append(mode_res)
    print(save_res)
    df = pd.DataFrame(save_res)
    df.to_excel('output.xlsx', index=False, header=False)

            # print(f'align {mode}: {sum(total_align)/len(total_align)}')
            # print(f'conf {mode}: {sum(total_conf)/len(total_conf)}')
            # print(f'info {mode}: {sum(total_info)/len(total_info)}')
            # print(f'safe {mode}: {sum(total_safe)/len(total_safe)}')
            # print(f'overcon {mode}: {sum(total_overc)/len(total_overc)}')
            # print(f'conserv {mode}: {sum(total_conserv)/len(total_conserv)}')

def extract_data():
    model_tail = {
        'llama2-chat-7b': 'llama7b',
        'llama3-8b-instruct': 'llama8b',
        'qwen2': 'qwen2',
        'llama2-chat-13b': 'llama13b'
    }
    for dataset in ['nq', 'hq']:
        for model in ['llama3-8b-instruct', 'qwen2', 'llama2-chat-13b']:
            for cnt in [2, 4, 6, 8]:
                if model == 'qwen2':
                    path = f'../share/res/{dataset}-mc/{model}/mid_layer/zero-shot-wo-gt-{cnt}-none-false-freeform-false-{model_tail[model]}/{dataset}-test-gene-choice-without-gt-{cnt}-none-False-freeform-False-qwen7b.jsonl'
                else:
                    path = f'../share/res/{dataset}-mc/{model}/mid_layer/zero-shot-wo-gt-{cnt}-none-false-freeform-false-{model_tail[model]}/{dataset}-test-gene-choice-without-gt-{cnt}-none-False-freeform-False-{model_tail[model]}.jsonl'
                data = read_json(path)
                new_data = [{'Res': item['Res'], 'has_answer': item['has_answer']} for item in data]
                out_file = f'../share/res/{dataset}-mc/{model}/extract/{dataset}-test-{cnt}.jsonl'
                if not os.path.exists(f'../share/res/{dataset}-mc/{model}/extract'):
                    os.makedirs(f'../share/res/{dataset}-mc/{model}/extract')
                write_jsonl(new_data, out_file)


if __name__ == '__main__':            
    rule_based_alignment_for_different_seed()

    


