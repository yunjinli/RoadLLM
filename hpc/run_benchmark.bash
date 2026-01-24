#!/bin/bash

# Configuration
IMAGE_FILE="/home/phd_li/git_repo/RoadLLM/test_images/urban.jpg" # Or change to "image.jpg" as you requested
PROMPT="Describe the image"
OUTPUT_JSON="comparison_results.json"

# Clear previous results
if [ -f "$OUTPUT_JSON" ]; then
    rm "$OUTPUT_JSON"
fi

echo "Starting Evaluation..."
echo "Image: $IMAGE_FILE"
echo "Prompt: $PROMPT"
echo "------------------------------------------------"

# --- Define Models (Extracted from job_cli.bash) ---

# 0.6B Model
echo "Running 0.6B Model..."
TRANSFORMERS_CACHE=/home/phd_li/.cache/huggingface/hub HF_HUB_OFFLINE=1 python -m llava.serve.evaluate_roadllm \
    --conv-mode qwen_3 \
    --model-alias "0.6B" \
    --model-base "/home/phd_li/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca" \
    --model-path "/home/phd_li/git_repo/RoadLLM/checkpoints/projectors/roadllm-llava-openai_clip-vit-large-patch14-336-Qwen_Qwen3-0.6B-mlp2x_gelu-pretrain-full-2gpus-5epoches" \
    --image-file "$IMAGE_FILE" \
    --prompt "$PROMPT" \
    --output-file "$OUTPUT_JSON"

# 1.7B Model
echo "Running 1.7B Model..."
TRANSFORMERS_CACHE=/home/phd_li/.cache/huggingface/hub HF_HUB_OFFLINE=1 python -m llava.serve.evaluate_roadllm \
    --conv-mode qwen_3 \
    --model-alias "1.7B" \
    --model-base "/home/phd_li/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e" \
    --model-path "/home/phd_li/git_repo/RoadLLM/checkpoints/projectors/roadllm-llava-openai_clip-vit-large-patch14-336-Qwen_Qwen3-1.7B-mlp2x_gelu-pretrain-full-2gpus-5epoches" \
    --image-file "$IMAGE_FILE" \
    --prompt "$PROMPT" \
    --output-file "$OUTPUT_JSON"

# 4B Model
echo "Running 4B Model..."
TRANSFORMERS_CACHE=/home/phd_li/.cache/huggingface/hub HF_HUB_OFFLINE=1 python -m llava.serve.evaluate_roadllm \
    --conv-mode qwen_3 \
    --model-alias "4B" \
    --model-base "/home/phd_li/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c" \
    --model-path "/home/phd_li/git_repo/RoadLLM/checkpoints/projectors/roadllm-llava-openai_clip-vit-large-patch14-336-Qwen_Qwen3-4B-mlp2x_gelu-pretrain-full-4gpus-5epoches" \
    --image-file "$IMAGE_FILE" \
    --prompt "$PROMPT" \
    --output-file "$OUTPUT_JSON"

# 8B Model
echo "Running 8B Model..."
TRANSFORMERS_CACHE=/home/phd_li/.cache/huggingface/hub HF_HUB_OFFLINE=1 python -m llava.serve.evaluate_roadllm \
    --conv-mode qwen_3 \
    --model-alias "8B" \
    --model-base "/home/phd_li/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218" \
    --model-path "/home/phd_li/git_repo/RoadLLM/checkpoints/projectors/roadllm-llava-openai_clip-vit-large-patch14-336-Qwen_Qwen3-8B-mlp2x_gelu-pretrain-full-4gpus-5epoches" \
    --image-file "$IMAGE_FILE" \
    --prompt "$PROMPT" \
    --output-file "$OUTPUT_JSON"

echo "Done! Results saved to $OUTPUT_JSON"