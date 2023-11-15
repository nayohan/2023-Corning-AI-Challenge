python test.py \
    --model_name_or_path princeton-nlp/sup-simcse-bert-base-uncased \
    --evaluation_file THUDM/webglm-qa \
    --data_type validation \
    --pooler_type cls \
    --max_length 64 \
    --gpu_id 0 \
    --top_k 1 \
    --use_fast_tokenizer

# python test.py \
#     --model_name_or_path bert-base-uncased \
#     --simcse_model_name_or_path "princeton-nlp/sup-simcse-bert-base-uncased"\
#     --evaluation_file THUDM/webglm-qa \
#     --data_type validation \
#     --do_simcse \
#     --gpu_id 0 \
#     --top_k 1 \
#     --use_fast_tokenizer
