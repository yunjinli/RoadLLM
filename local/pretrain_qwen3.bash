LLM_VERSION="Qwen/Qwen3-0.6B"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="openai/clip-vit-large-patch14-336"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

############### Pretrain ################

PROMPT_VERSION=qwen_3

BASE_RUN_NAME="roadllm-llava-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-mlp2x_gelu-pretrain"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

command="python llava/train/train_mem.py \
    --model_name_or_path ${LLM_VERSION} \
    --version ${PROMPT_VERSION} \
    --data_path  ./configs/pretrained.json \
    --image_folder /home/yunjinli/git_repo/RoadLLM/dataset_subset \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_tunable_parts="mm_mlp_adapter" \
    --mm_vision_select_layer -2 \
    --mm_projector_type mlp2x_gelu \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --output_dir ./checkpoints/projectors/${BASE_RUN_NAME}-full-${NUM_GPUS}gpus \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --fp16 False \
    --bf16 True \
    --save_strategy "steps" \
    --save_steps 100 \
    --learning_rate 2e-3 \
    --max_grad_norm 1.0 \
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
eval "${command}"
