Drafter_NAME="vicuna-68m"
Model_NAME="vicuna-7b-v1.3"
temp=0.1
# device_name="orin_30w"
# for num_assistant_tokens in 1 2 3 4 5 10 15 20 25
# do 
#     python process/calculate_theory_speed.py --file-path /home/jingcan/workspace/Spec-Bench/data/orin_50w_spec_bench/model_answer/${Model_NAME}-sps-${Drafter_NAME}-float16-temp-${temp}-asstkn-${num_assistant_tokens}.jsonl  --base-path /home/jingcan/workspace/Spec-Bench/data/orin_50w_spec_bench/model_answer/${Model_NAME}-vanilla-float16-temp-${temp}.jsonl --tokenizer-path lmsys/${Model_NAME} --device-name ${device_name} > /home/jingcan/workspace/Spec-Bench/data/${device_name}_spec_bench/speed/speed_${Model_NAME}-sps-${Drafter_NAME}-float16-temp-${temp}-asstkn-${num_assistant_tokens}.log
# done
device_name="cloud"
for num_assistant_tokens in 1 2 3 4 5 10 15 20 25
do 
    python evaluation/speed.py --file-path /home/jingcan/workspace/Spec-Bench/data/cloud_spec_bench/model_answer/${Model_NAME}-sps-${Drafter_NAME}-float16-temp-${temp}-asstkn-${num_assistant_tokens}.jsonl  --base-path /home/jingcan/workspace/Spec-Bench/data/cloud_spec_bench/model_answer/${Model_NAME}-vanilla-float16-temp-0.0.jsonl --tokenizer-path lmsys/${Model_NAME} > /home/jingcan/workspace/Spec-Bench/data/${device_name}_spec_bench/speed/speed_${Model_NAME}-sps-${Drafter_NAME}-float16-temp-0.1-asstkn-${num_assistant_tokens}.log
done