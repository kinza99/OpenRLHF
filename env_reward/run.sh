eval_path='/home/hesiyang/DAMO-ConvAI/bird/llm/data/dev.json'
dev_path='./output/'
db_root_path='/home/hesiyang/DAMO-ConvAI/bird/llm/data/dev_databases/'
use_knowledge='True'
not_use_knowledge='False'
mode='dev' # choose dev or dev
cot='True'
no_cot='False'
data_mode='dev'

YOUR_API_KEY=''

engine1='code-davinci-002'
engine2='text-davinci-003'
engine3='gpt-3.5-turbo'

# data_output_path='./exp_result/gpt_output/'
# data_kg_output_path='./exp_result/gpt_output_kg/'

# data_output_path='./exp_result_testfromscratch/turbo_output/'
# data_kg_output_path='./exp_result_testfromscratch/turbo_output_kg/'

ground_truth_path='/home/hesiyang/DAMO-ConvAI/bird/llm/data/'
result_output_file='/home/hesiyang/DAMO-ConvAI/bird/result_ouput'

echo 'generate GPT3.5 batch without knowledge'
python3 -u /home/hesiyang/DAMO-ConvAI/bird/env_reward_sys/run.py --db_root_path ${db_root_path} --api_key ${YOUR_API_KEY} --mode ${mode} \
--engine ${engine3} --eval_path ${eval_path} --use_knowledge ${use_knowledge} \
--chain_of_thought ${no_cot} --ground_truth_path ${ground_truth_path} --data_mode ${data_mode} --result_output_file ${result_output_file}