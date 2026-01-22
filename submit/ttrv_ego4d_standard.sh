#!/bin/bash -l

#SBATCH -o /dais/fs/scratch/dduka/logs/verl/verl_standard.out
#SBATCH -e /dais/fs/scratch/dduka/logs/verl/verl_standard.err

#SBATCH -J verl_standard
#SBATCH --time=01:59:59

#SBATCH --nodes=1
#SBATCH --partition="gpu1"
#SBATCH --cpus-per-task=48
#SBATCH --threads-per-core=1

#SBATCH --gres=gpu:h200:4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=1000000

# #SBATCH --array=0-2%1

micromamba activate verl
module load cuda/12.8
module load gcc/14

cd /u/dduka/project/RL/TTRV/verl

unset VLLM_ATTENTION_BACKEND
export VLLM_USE_V1=1
export PYTHONPATH=$PYTHONPATH:/u/dduka/project/RL/TTRV/verl

mkdir -p logs

DATE=$(date +%m%d)
TIME_TAG=$(date +%H%M%S)

TASK="tag"
NO_GPU=4
EPISODE=2
ADVANTAGE="grpo"

MAX_PROMPT_LENGTH=7524
MAX_RESPONSE_LENGTH=$((64 * 1))

N=10 #This sets the number of samples generated during validation: 

DATA_TRAIN_BATCH_SIZE=4 # Batch size
N_VOTES_PER_PROMPT=16 # Total responses generated per prompt
N_SAMPLES_PER_PROMPT=10 # Number of responses kept for the PPO training: used in self._select_top_k_per_prompt(...)
MINI_BATCH_SIZE=1
MICRO_BATCH_SIZE=2

TRAIN_FILE="/u/dduka/project/RL/TTRV/verl/data/tag/standard/train.parquet"
VAL_FILE="/u/dduka/project/RL/TTRV/verl/data/tag/standard/test.parquet"

DATA_LOCAL_DIR="/u/dduka/project/RL/TTRV/verl/data"

BACKBONE_PATH="Qwen/Qwen3-VL-8B-Instruct"

MODEL="${TASK}-${BACKBONE_PATH}"
EXPERIMENT="TTRL-EGO4D-TAR-ORIGINAL-SEGMENTS"

WANDB_PROJECT="TTRL-verl"
LOG_NAME="${EXPERIMENT}-${MODEL}-${ADVANTAGE}"
OUTPUT_DIR="/dais/fs/scratch/dduka/training_metadata/ttrv/checkpoints/${WANDB_PROJECT}/${MODEL}/${EXPERIMENT}-${ADVANTAGE}"

BACKBONE_SAFE=$(echo "$BACKBONE_PATH" | tr '/' '_')
LOG_FILE="/dais/fs/scratch/dduka/logs/ttrv/${EXPERIMENT}_${BACKBONE_SAFE}_${EPISODE}e.log"

python verl/trainer/main_ppo.py \
  reward_model.reward_manager=ttrl \
  reward_model.reward_kwargs.n_samples_per_prompt=$N_SAMPLES_PER_PROMPT \
  reward_model.reward_kwargs.n_votes_per_prompt=$N_VOTES_PER_PROMPT \
  reward_model.reward_kwargs.mode="train" \
  data.max_prompt_length=$MAX_PROMPT_LENGTH \
  data.max_response_length=$MAX_RESPONSE_LENGTH \
  data.train_batch_size=$DATA_TRAIN_BATCH_SIZE \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  data.train_files=["$TRAIN_FILE"] \
  data.val_files=["$VAL_FILE"] \
  actor_rollout_ref.model.path=$BACKBONE_PATH \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.optim.lr=5e-7 \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.03 \
  actor_rollout_ref.actor.optim.warmup_style='cosine' \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.free_cache_engine=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.65 \
  actor_rollout_ref.rollout.do_vote=True \
  actor_rollout_ref.rollout.n_vote=$N_VOTES_PER_PROMPT \
  actor_rollout_ref.rollout.n=$N_SAMPLES_PER_PROMPT \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.rollout.val_kwargs.n=$N \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
  actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
  actor_rollout_ref.rollout.max_model_len=32768 \
  actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
  critic.optim.lr=9e-6 \
  critic.model.use_remove_padding=True \
  critic.model.path=$BACKBONE_PATH \
  critic.model.enable_gradient_checkpointing=True \
  critic.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
  critic.model.fsdp_config.param_offload=False \
  critic.model.fsdp_config.optimizer_offload=False \
  algorithm.kl_ctrl.kl_coef=0.00 \
  algorithm.adv_estimator=$ADVANTAGE \
  trainer.logger=['console','wandb'] \
  trainer.project_name=$WANDB_PROJECT \
  trainer.experiment_name=$LOG_NAME \
  trainer.n_gpus_per_node=$NO_GPU \
  trainer.nnodes=1 \
  trainer.val_before_train=False \
  trainer.save_freq=200 \
  trainer.test_freq=200 \
  trainer.max_actor_ckpt_to_keep=3 \
  trainer.max_critic_ckpt_to_keep=3 \
  trainer.default_local_dir=$OUTPUT_DIR \
  trainer.total_epochs=$EPISODE "$@" 2>&1 | tee "$LOG_FILE"