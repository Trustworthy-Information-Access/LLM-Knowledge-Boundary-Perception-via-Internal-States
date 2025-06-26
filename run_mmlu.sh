#!/bin/bash



# 第一部分：执行 run_nq.py 脚本
# need_layers=mid
# for task in nq hq
# do
#     for dataset in test dev train
#     do
#         for model_path in "../models/llama3_8b_instruct" "../models/llama2-7B-chat"
#         do
#             if [ "$model_path" == "../models/llama3_8b_instruct" ]; then
#                 outfile="./res/${task}/${task}_${dataset}_llama8b_tokens_cot_${need_layers}_layer.jsonl"
#             elif [ "$model_path" == "../models/llama2-7B-chat" ]; then
#                 outfile="./res/${task}/${task}_${dataset}_llama7b_tokens_cot_${need_layers}_layer.jsonl"
#             fi
#             python -u run_nq.py \
#                 --source ../share/datasets/${task}/${task}-${dataset}.jsonl \
#                 --type qa_cot \
#                 --ra none \
#                 --outfile $outfile \
#                 --model_path $model_path \
#                 --batch_size 18 \
#                 --task nq \
#                 --max_new_tokens 256 \
#                 --hidden_states 1 \
#                 --hidden_idx_mode first,last,avg \
#                 --need_layers $need_layers
#         done
#     done
# done

# 设置第二部分的变量
# type=mc_qa_evidence
# task=mmlu
# source=../datasets/mmlu/
# need_layers=mid
# # 第二部分：执行 run_mmlu.py 脚本
# for model_path in "../models/llama3_8b_instruct" "../models/llama2-7B-chat"
# do
#     if [ "$model_path" == "../models/llama3_8b_instruct" ]; then
#         outfile="./res/mmlu/llama3_8b_instruct/zero-shot-$need_layers-explain/"
#     elif [ "$model_path" == "../models/llama2-7B-chat" ]; then
#         outfile="./res/mmlu/llama2-chat-7b/zero-shot-$need_layers-explain/"
#     fi

#     python run_mmlu.py \
#         --source $source \
#         --type $type \
#         --ra none \
#         --outfile $outfile \
#         --n_shot 0 \
#         --model_path $model_path \
#         --batch_size 4 \
#         --task $task \
#         --max_new_tokens 128 \
#         --hidden_states 1 \
#         --need_layers $need_layers \
#         --hidden_idx_mode first,last,avg,min,ans,dim_min,dim_max
# done

#run qwen2

need_layers=mid
model_path=../models/Qwem2-7B-Instruct
for task in nq hq
do
    for dataset in test dev train
    do
        outfile="./res/${task}/${task}_${dataset}_llama7b_tokens_cot_${need_layers}_layer.jsonl"
        python -u run_nq.py \
            --source ../share/datasets/${task}/${task}-${dataset}.jsonl \
            --type qa \
            --ra none \
            --outfile $outfile \
            --model_path $model_path \
            --batch_size 36 \
            --task nq \
            --max_new_tokens 64 \
            --hidden_states 1 \
            --hidden_idx_mode first,last,avg \
            --need_layers $need_layers
    done
done