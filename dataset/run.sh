OPENAI_API_KEY=''

python main.py \
    --model_path_or_name gpt-3.5-turbo-instruct \
    --input_path ./paperDB \
    --output_path ./dataset.jsonl \
    --output_dir_retriever ../retriever/codes/data/train.jsonl \
    --output_dir_generator ../data/dqa.jsonl \
    --chunk_size 256 \
    --chunk_overlap 20 \
    --chunk_mode recursive \
    --extension pdf \
    --doc_length 200 \
    --max_length 512 \
    --openai_api_key ${OPENAI_API_KEY}
