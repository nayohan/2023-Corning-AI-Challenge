#! /usr/bin/env bash

#SBATCH -p  normal
#SBATCH --gres=gpu:1

export GLOO_SOCKET_IFNAME=eth0

maindir=$1
datadir=${maindir}data
codedir=${maindir}codes
taskdir=${datadir}/tasks

NODE_NUM=1
INDEX=0
MASTER_ADDR="127.0.0.1"
MASTER_PORT=12321
GPU_NUM_PER_NODE=4
RAYGPUS=1

MAXLEN=2048
EPOCH=2
# tasks=("DG" "DHG" "DS")
# tasks=("DocQA")
tasks="DocQA7_MS"
# models=("t5-small" "t5-base" "t5-large" "t5-3b")
# models=("flan-t5-small" "flan-t5-base" "flan-t5-large" "flan-t5-3b")
# models=("llama-7b" "llama-13b" "llama-33b")
# models=("t5-small" "t5-base" "t5-large" "flan-t5-small" "flan-t5-base" "flan-t5-large"  "t5-3b"  "flan-t5-3b")
# models=("llama-7b" "llama-13b")
# models="kollama-13b"
# models="llama-7b"
# models=("meta-llama/Llama-2-13b-chat-hf" "upstage/SOLAR-10.7B-v1.0")
# models=("allenai/OLMo-7B")
# models=("mistralai/Mixtral-8x7B-Instruct-v0.1") 
models=("lmsys/vicuna-7b-v1.5" "lmsys/vicuna-13b-v1.5")

for model in "${models[@]}"
    do
    # raw_model_path=${maindir}pretrained_model/${model}/
    raw_model_path=${model}

    # tuning
    for task in "${tasks[@]}"
        do
        test_data=${taskdir}/${task}/test.jsonl

        # zeroshot inference on one node
        # python3 ${codedir}/eval/get_model_infer_simple.py \
        # --model-id ${model}_zeroshot \
        # --model-path ${raw_model_path} \
        # --question-file ${test_data} \
        # --answer-file ${datadir}/instruction_testing/inf_${model}_${task}_zeroshot.jsonl \
        # --num-gpus $GPU_NUM_PER_NODE \
        # --ray-num-gpus ${RAYGPUS}

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
            "llama-7b"|"lmsys/vicuna-7b-v1.5"|"allenai/OLMo-7B")
                PER_GPU_BATCH=2
                GRA_ACC=8
                DS_CONFIG="7b"
                ;;
            "t5-11b"|"flan-t5-11b"|"meta-llama/Llama-2-13b-chat-hf"|"kollama-13b"|\
            "upstage/SOLAR-10.7B-v1.0"|"lmsys/vicuna-13b-v1.5"|"mistralai/Mixtral-8x7B-Instruct-v0.1")
                PER_GPU_BATCH=2
                GRA_ACC=8
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
        python3 ${codedir}/train/data_preprocess_task.py \
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
            ${codedir}/train/train.py \
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
            --eval_steps=5 \
            --logging_steps=5 \
            --max_eval_samples=1000 \
            --run_name="${model}_${task}_e${EPOCH}"

        # # # # tuning inference
        python3 ${codedir}/eval/get_model_infer_simple.py \
            --model-id ${model}_${task} \
            --model-path ${model_output_path} \
            --question-file ${test_data} \
            --answer-file ${datadir}/instruction_testing/inf_${model}_${task}.jsonl \
            --num-gpus $GPU_NUM_PER_NODE \
            --ray-num-gpus ${RAYGPUS}
        done
    done