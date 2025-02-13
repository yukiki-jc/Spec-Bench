Drafter_NAME="Qwen2.5-0.5B-Instruct"
Model_NAME="Qwen2.5-7B-Instruct"
python evaluation/speed.py --file-path /home/jingcan/workspace/Spec-Bench/data/spec_bench/model_answer/${Model_NAME}-sps-${Drafter_NAME}-float16-temp-0.0.jsonl  --base-path /home/jingcan/workspace/Spec-Bench/data/spec_bench/model_answer/${Model_NAME}-vanilla-float16-temp-0.0.jsonl --tokenizer-path Qwen/${Model_NAME}