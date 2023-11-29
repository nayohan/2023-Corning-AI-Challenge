#! /usr/bin/env bash

#SBATCH -p  normal
#SBATCH --gres=gpu:1

export GLOO_SOCKET_IFNAME=eth0

maindir=$1
datadir=${maindir}data
codedir=${maindir}code
taskdir=${datadir}/tasks

NODE_NUM=1
INDEX=0
MASTER_ADDR="127.0.0.1"
MASTER_PORT=12321
GPU_NUM_PER_NODE=1

MAXLEN=2048
EPOCH=2
# tasks=("DG" "DHG" "DS")
# tasks=("DocQA")
tasks="DocQA2"
# models=("t5-small" "t5-base" "t5-large" "t5-3b")
# models=("flan-t5-small" "flan-t5-base" "flan-t5-large" "flan-t5-3b")
# models=("llama-7b" "llama-13b" "llama-33b")
# models=("t5-small" "t5-base" "t5-large" "flan-t5-small" "flan-t5-base" "flan-t5-large"  "t5-3b"  "flan-t5-3b")
# models=("llama-7b" "llama-13b")
models="kollama-13b"
# models="llama-7b"

for model in "${models[@]}"
    do
    raw_model_path=${maindir}pretrained_model/${model}/
    case ${model} in 
        "t5-11b"|"llama-13b"|"llama-33b"|"kollama-13b")
            RAYGPUS=1
            ;;
        "t5-small"|"t5-base"|"t5-large"|"t5-3b"|\
        "flan-t5-small"|"flan-t5-base"|"flan-t5-large"|"flan-t5-3b"|\
        "llama-7b")
            RAYGPUS=1
            ;;
    esac
    
    # tuning
    for task in "${tasks[@]}"
        do
        test_data=${taskdir}/${task}/test.jsonl

        # zeroshot inference on one node
        python3 ${codedir}/codes/eval/get_model_infer_simple.py \
        --model-id ${model}_zeroshot \
        --model-path ${raw_model_path} \
        --question-file ${test_data} \
        --answer-file ${datadir}/instruction_testing/inf_${model}_${task}_zeroshot.jsonl \
        --num-gpus $GPU_NUM_PER_NODE \
        --ray-num-gpus ${RAYGPUS}

        case ${model} in 
            "t5-small"|"t5-base"|"t5-large"|\
            "flan-t5-small"|"flan-t5-base"|"flan-t5-large")
                PER_GPU_BATCH=16
                GRA_ACC=1
                DS_CONFIG="1b"
                ;;
            "t5-3b"|"flan-t5-3b")
                PER_GPU_BATCH=16
                GRA_ACC=1
                DS_CONFIG="7b"
                ;;
            "llama-7b")
                PER_GPU_BATCH=8
                GRA_ACC=2
                DS_CONFIG="7b"
                ;;
            "t5-11b"|"flan-t5-11b"|"llama-13b"|"kollama-13b")
                PER_GPU_BATCH=4
                GRA_ACC=4
                DS_CONFIG="13b"
                ;;
            "llama-33b")
                PER_GPU_BATCH=4
                GRA_ACC=4
                DS_CONFIG="33b"
                ;;
        esac

        data_path=${taskdir}/${task}
        preprocessed_data_dir=${taskdir}/processed/${task}/processed_${task}_${model%-*}.pt
        model_output_path=${maindir}model/${model}_${task}/
        deepspeed_config_path=${codedir}/configs/ds_config_${DS_CONFIG}.json

        # # # # # train data preprocess
        python3 ${codedir}/codes/train/data_preprocess_task.py \
            --model_name_or_path ${raw_model_path} \
            --data_path ${data_path} \
            --preprocessing_num_workers=1 \
            --model_max_length ${MAXLEN} \
            --preprocessed_path ${preprocessed_data_dir}
        
        # # # # training: avaliable for multi nodes
        torchrun --nnodes=$NODE_NUM \
            --node_rank=$INDEX \
            --nproc_per_node $GPU_NUM_PER_NODE \
            --master_addr $MASTER_ADDR \
            --master_port $MASTER_PORT \
            ${codedir}/codes/train/train.py \
            --model_name_or_path ${raw_model_path} \
            --bf16 True \
            --output_dir ${model_output_path} \
            --num_train_epochs ${EPOCH} \
            --per_device_train_batch_size ${PER_GPU_BATCH} \
            --gradient_accumulation_steps ${GRA_ACC} \
            --save_strategy "steps" \
            --save_steps 1500 \
            --save_total_limit 1 \
            --learning_rate 2e-5 \
            --log_level "info" \
            --logging_strategy "steps" \
            --logging_steps 1 \
            --weight_decay 0. \
            --warmup_ratio 0.04 \
            --lr_scheduler_type "cosine" \
            --deepspeed ${deepspeed_config_path} \
            --tf32 True \
            --model_max_length ${MAXLEN} \
            --preprocessed_path ${preprocessed_data_dir} \
            --gradient_checkpointing True \
            --logging_first_step \
            --do_eval \
            --evaluation_strategy='steps' \
            --eval_steps=10 \
            --logging_steps=10 \
            --max_eval_samples=1000 \
            --run_name="${model}_${task}_e${EPOCH}"
        
        # # # # tuning inference
        python3 ${codedir}/codes/eval/get_model_infer_simple.py \
            --model-id ${model}_${task} \
            --model-path ${model_output_path} \
            --question-file ${test_data} \
            --answer-file ${datadir}/instruction_testing/inf_${model}_${task}.jsonl \
            --num-gpus $GPU_NUM_PER_NODE \
            --ray-num-gpus ${RAYGPUS}
        done
    done