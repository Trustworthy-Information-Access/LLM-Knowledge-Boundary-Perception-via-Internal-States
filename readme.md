This is the official code for the paper *[Towards Fully Exploiting LLM Internal States to Enhance Knowledge Boundary Perception](https://arxiv.org/abs/2502.11677)*(**ACL2025 Main**). The repository is currently in progress.

## Intro

This repo is used for decoder-only LMs' inference and the code is based on transformers. **We fully utilized the Transformers library without using any inference frameworks.** The repository is continuously updated.

- Support batch_generate
- Return `scores`/`hidden_states` for the generated tokens
- Supporting task format: free-form generation; multi-choice qa
- Currently, only single-GPU inference is supported.

## Usage

### Free-form Generation
```bash
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
```
- `--hidden_states`: If you want to obtain the hidden state information related to the generated tokens, you need to specify this parameter; otherwise, remove it.
- `--hidden_idx_mode`: We support [`first,last,avg,min,every']
  - `first, last`: Obtain the hidden state at the specified layers for the **first** or **last** token during generation. 
  - `avg`: Obtain the average hidden state across all the generated tokens at the specified layers.
  - `min`: Obtain the hidden state of the generated token with min probability at the specified layers.
- `--need_layers`: We support [`all,last,mid`], which specify all the layers, the last layer, the mid layer (16 for 32-layer models) for getting hidden state information.


### Multi-Choice Generation

```bash
python run_mmlu.py \
    --source YOUR_DATA_PATH \
    --type qa \
    --ra none \
    --outfile YOUR_OUT_FILE_PATH \
    --n_shot 0 \
    --model_path YOUR_MODEL_PATH \
    --batch_size 16 \
    --task mmlu \
    --max_new_tokens 64 \
    --hidden_states 1 \
    --need_layers mid
```
- Very similar to `Free-Form Generation`.
- We find the choices [`A,B,C,D`] in the response and get related information, so there is no need to specify `--hidden_idx_mode`
## Note
- `--hidden_idx_mode, --need_layers` only works when you specify `--hidden_states 1`

We support only `llama2-chat` and `llama3-instruct` series models because each llm need its own prompt format. You can add more prompt templates in `utils/prompt.py` to support more LLMs.
- `llama3-8b-instruct` use `'<|eot_id|>'` instead of `<eos>` to represent the end of generation.

## Example
Free-form generation
```bash
python -u run_nq.py \
    --source ./data/nq/nq-dev.jsonl \
    --type qa \
    --ra none \
    --outfile ./data/nq/nq-dev-res.jsonl \
    --model_path YOUR_MODEL_PATH \
    --batch_size 36 \
    --task nq \
    --max_new_tokens 64 \
```

Multi-choice generation
```bash
python run_mmlu.py \
    --source ./data/truthfulqa \
    --type qa \
    --ra none \
    --outfile ./data/truthfulqa/res/ \
    --n_shot 0 \
    --model_path YOUR_MODEL_PATH \
    --batch_size 16 \
    --task tq \
    --max_new_tokens 64 \
```

## Paper Implementation
- QA
  - We ask the model to answer questions. Necessary scripts can be found in `run_mmlu.sh`(for multi-choice QA) and `run_nq.sh`(for free-form generation). The core file is `./utils/llm.py`
- MLP Training
  - Data preprocess: convert files from QA to `.pt` data for training. See `./hidden_state_detection/data.py`
    - If you want to sample training data for balanced classes(acc=1/0), see `sample_training_data` in `./hidden_state_detection/data.py`
  - Training:
    - Necessary scripts can be found in `./hidden_state_detection/scripts` like `run_nq.sh`, `run_nq_mc.sh`, `run_nq_mc_sample.sh`. You can design the scripts.
- $C^3$
  - Question Reformulation: 
    - Run `run_nq.sh` and specify `--type qa_more`
    - Construct multi-choice questions. See `./mc_data/data.py`
  - MLP Training: The same as the above.
