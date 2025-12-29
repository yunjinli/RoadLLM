# export OMP_NUM_THREADS=8
# export NCCL_IB_DISABLE=0
# export NCCL_IB_GID_INDEX=3
# export NCCL_SOCKET_IFNAME=eth0
# export NCCL_DEBUG=INFO

LLM_VERSION="Qwen/Qwen3-0.6B"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="openai/clip-vit-large-patch14-336"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

############### Pretrain ################

PROMPT_VERSION=qwen_3

BASE_RUN_NAME="roadllm-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-mlp2x_gelu-pretrain"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

# ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \
#     llava/train/train_mem.py \
#     --deepspeed scripts/zero3.json \
#     --model_name_or_path ${LLM_VERSION} \
#     --version ${PROMPT_VERSION} \
#     --data_path /blip_558k/blip_558k_plain.json \
#     --image_folder /blip_558k/images \
#     --vision_tower ${VISION_MODEL_VERSION} \
#     --mm_tunable_parts="mm_mlp_adapter" \
#     --mm_vision_select_layer -2 \
#     --mm_projector_type mlp2x_gelu \
#     --mm_use_im_start_end False \
#     --mm_use_im_patch_token False \
#     --bf16 True \
#     --output_dir /checkpoints/projectors/${BASE_RUN_NAME} \
#     --num_train_epochs 1 \
#     --per_device_train_batch_size 16 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 1 \
#     --evaluation_strategy "no" \
#     --save_strategy "no" \
#     --save_steps 50000 \
#     --learning_rate 1e-3 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length 8192 \
#     --gradient_checkpointing True \
#     --dataloader_num_workers 16 \
#     --lazy_preprocess True \
#     --report_to wandb \
#     --run_name $BASE_RUN_NAME \
#     --attn_implementation sdpa



command="TRANSFORMERS_CACHE=/home/phd_li/.cache/huggingface/hub HF_HUB_OFFLINE=1 python llava/train/train_mem.py \
    --model_name_or_path ${LLM_VERSION} \
    --version ${PROMPT_VERSION} \
    --data_path  ./pretrained_subset.json \
    --image_folder ./dataset_subset \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_tunable_parts="mm_mlp_adapter" \
    --mm_vision_select_layer -2 \
    --mm_projector_type mlp2x_gelu \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --output_dir ./checkpoints/projectors/${BASE_RUN_NAME} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --fp16 True \
    --save_strategy "no" \
    --save_steps 1000 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name $BASE_RUN_NAME \
    --attn_implementation sdpa \
    --gradient_checkpointing False \
    "

# You can delete the sdpa attn_implementation if you want to use flash attn

bsub -Is -q gpu -gpu "num=1:j_exclusive=yes:gmem=40G" -R "select[type==X64LIN]" eval "${command}"