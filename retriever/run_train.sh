MODEL_PATH=''
TRAIN_PATH=''
EVAL_PATH=''

python main.py \
    --output_path ./codes/result/model.pt \
    --train_fn ${TRAIN_PATH} \
    --valid_fn ${EVAL_PATH} \
    --model_path_or_name ${MODEL_PATH} \
    --batch_size 128 \
    --eval_step 125 \
    --lr 3e-5 \
    --n_epochs 3 \
    --max_length 128 \
    --pooler_type cls \
    --temp 0.05 \
    --fp16 \