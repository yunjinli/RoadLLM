if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <num_gpus> <num_cpus> <llm> <epoches>"
    exit 1
fi

NUM_GPUS="$1"
NUM_CPUS="$2"
LLM_VERSION="$3"
NUM_EPOCHS="$4"
# LLM_VERSION="Qwen/Qwen3-0.6B"

LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="openai/clip-vit-large-patch14-336"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"


############### Pretrain ################

PROMPT_VERSION=qwen_3

BASE_RUN_NAME="roadllm-llava-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-mlp2x_gelu-pretrain"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

# echo "Available GPUs:"
# nvidia-smi --list-gpus
# echo ""

command="CC=/opt/gcc_9.2.0/bin/gcc CXX=/opt/gcc_9.2.0/bin/g++ WANDB_MODE=offline TRANSFORMERS_CACHE=/home/phd_li/.cache/huggingface/hub HF_HUB_OFFLINE=1 deepspeed --num_gpus ${NUM_GPUS} llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path ${LLM_VERSION} \
    --version ${PROMPT_VERSION} \
    --data_path  ./configs/pretrained.json \
    --image_folder /home/phd_li/dataset \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_tunable_parts="mm_mlp_adapter" \
    --mm_vision_select_layer -2 \
    --mm_projector_type mlp2x_gelu \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --output_dir ./checkpoints/projectors/${BASE_RUN_NAME}-full-${NUM_GPUS}gpus-${NUM_EPOCHS}epoches \
    --num_train_epochs ${NUM_EPOCHS} \
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
    --dataloader_num_workers ${NUM_CPUS} \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name $BASE_RUN_NAME \
    --attn_implementation sdpa \
    --gradient_checkpointing False \
    "

# You can delete the sdpa attn_implementation if you want to use flash attn
# eval "${command}"

TOTAL_CPUS=$(expr ${NUM_GPUS} \* ${NUM_CPUS})
bsub -Is -q gpu -gpu "num=${NUM_GPUS}:j_exclusive=yes:gmem=40G" -n ${TOTAL_CPUS} -R "select[type==X64LIN]" eval "nvidia-smi --list-gpus & ${command}"