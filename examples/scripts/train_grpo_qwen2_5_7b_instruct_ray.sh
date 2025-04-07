set -x

# /cpfs01/shared/llm_ddd/puyu_transfer_data/guohonglin/hf_hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/bb46c15ee4bb56c5b63245ef50fd7637234d6f75_no_yarn
# /cpfs01/shared/llm_ddd/puyu_transfer_data/guohonglin/hf_hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/2fd50615a2a9792d223eba8e0741aa90ef21a869

# ray job submit --address="http://127.0.0.1:8265" \
#    --runtime-env-json='{"working_dir": "/cpfs01/user/duhe/OpenRLHF"}' \
#    -- 
   # --colocate_actor_ref \
python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 2 \
   --reward_num_nodes 1 \
   --reward_num_gpus_per_node 1 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 4 \
   --vllm_num_engines 2 \
   --vllm_tensor_parallel_size 1 \
   --vllm_enable_sleep \
   --pretrain /cpfs01/shared/llm_ddd/puyu_transfer_data/guohonglin/hf_hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/bb46c15ee4bb56c5b63245ef50fd7637234d6f75_no_yarn \
   --remote_rm_url ./env_reward/env_reward.py \
   --save_path /cpfs01/shared/llm_code/duhe/agent_rl_models/qwen2-5-7b-instruct-grpo \
   --micro_train_batch_size 2 \
   --train_batch_size 64 \
   --micro_rollout_batch_size 4 \
   --rollout_batch_size 128 \
   --enable_prefix_caching \
   --n_samples_per_prompt 8 \
   --max_epochs 1 \
   --prompt_max_len 31744 \
   --max_samples 100000 \
   --generate_max_len 1024 \
   --init_kl_coef 1e-3 \
   --gamma 1.0 \
   --use_kl_loss \
   --kl_estimator k3 \
   --advantage_estimator group_norm \
   --zero_stage 3 \
   --bf16 \
   --actor_learning_rate 1e-7 \
   --prompt_data data/bird_train.jsonl \
   --input_key message \
   --label_key label \
   --apply_chat_template \
   --normalize_reward \
   --adam_offload \
   --gradient_checkpointing \
   --packing_samples \
   --save_steps 50 \
   --max_turns 1 \
   --adam_offload \
   --overlap_comm \
   --use_wandb b2235b6e414a6ebd3ff7fa3162fbce411a42b5ad \
   --wandb_project qwen2-5-7b-instruct-grpo-bird-max_turns3 \
   --num_episodes 5
# You could also try
#   --kl_estimator k2 \

# also supports --advantage_estimator rloo | reinforce_baseline
