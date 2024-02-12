MODEL_PATH=''
TEST_PATH=''

python codes/test.py \
    --model_name_or_path ${MODEL_PATH} \
    --evaluation_file ${TEST_PATH} \
    --pooler_type cls \
    --max_length 64 \
    --gpu_id 0 \
    --top_k 5 \
