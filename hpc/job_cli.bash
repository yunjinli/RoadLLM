if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <test_image_name>"
    exit 1
fi

IMAGE_NAME="$1"
## 8B-1epochs
# command="TRANSFORMERS_CACHE=/home/phd_li/.cache/huggingface/hub HF_HUB_OFFLINE=1 python -m llava.serve.cli_roadllm --conv-mode qwen_3 --model-base /home/phd_li/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218 --model-path /home/phd_li/git_repo/RoadLLM/checkpoints/projectors/roadllm-llava-openai_clip-vit-large-patch14-336-Qwen_Qwen3-8B-mlp2x_gelu-pretrain-full-4gpus --image-file /home/phd_li/git_repo/RoadLLM/test_images/urban.jpg"
## 8B-5epochs
# command="TRANSFORMERS_CACHE=/home/phd_li/.cache/huggingface/hub HF_HUB_OFFLINE=1 python -m llava.serve.cli_roadllm --conv-mode qwen_3 --model-base /home/phd_li/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218 --model-path /home/phd_li/git_repo/RoadLLM/checkpoints/projectors/roadllm-llava-openai_clip-vit-large-patch14-336-Qwen_Qwen3-8B-mlp2x_gelu-pretrain-full-4gpus-5epoches --image-file /home/phd_li/git_repo/RoadLLM/test_images/urban.jpg"
## 0.6B-5epochs
# command="TRANSFORMERS_CACHE=/home/phd_li/.cache/huggingface/hub HF_HUB_OFFLINE=1 python -m llava.serve.cli_roadllm --conv-mode qwen_3 --model-base /home/phd_li/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca --model-path /home/phd_li/git_repo/RoadLLM/checkpoints/projectors/roadllm-llava-openai_clip-vit-large-patch14-336-Qwen_Qwen3-0.6B-mlp2x_gelu-pretrain-full-2gpus-5epoches --image-file /home/phd_li/git_repo/RoadLLM/test_images/urban.jpg"
## 1.7B-5epochs
# command="TRANSFORMERS_CACHE=/home/phd_li/.cache/huggingface/hub HF_HUB_OFFLINE=1 python -m llava.serve.cli_roadllm --conv-mode qwen_3 --model-base /home/phd_li/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e --model-path /home/phd_li/git_repo/RoadLLM/checkpoints/projectors/roadllm-llava-openai_clip-vit-large-patch14-336-Qwen_Qwen3-1.7B-mlp2x_gelu-pretrain-full-2gpus-5epoches --image-file /home/phd_li/git_repo/RoadLLM/test_images/urban.jpg"
## 1.7B-1epochs
# command="TRANSFORMERS_CACHE=/home/phd_li/.cache/huggingface/hub HF_HUB_OFFLINE=1 python -m llava.serve.cli_roadllm --conv-mode qwen_3 --model-base /home/phd_li/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e --model-path /home/phd_li/git_repo/RoadLLM/checkpoints/projectors/roadllm-llava-openai_clip-vit-large-patch14-336-Qwen_Qwen3-1.7B-mlp2x_gelu-pretrain-full-4gpus --image-file /home/phd_li/git_repo/RoadLLM/test_images/urban.jpg"
## 4B-1epochs
# command="TRANSFORMERS_CACHE=/home/phd_li/.cache/huggingface/hub HF_HUB_OFFLINE=1 python -m llava.serve.cli_roadllm --conv-mode qwen_3 --model-base /home/phd_li/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c --model-path /home/phd_li/git_repo/RoadLLM/checkpoints/projectors/roadllm-llava-openai_clip-vit-large-patch14-336-Qwen_Qwen3-4B-mlp2x_gelu-pretrain-full-4gpus --image-file /home/phd_li/git_repo/RoadLLM/test_images/urban.jpg"
## 4B-5epochs
command="TRANSFORMERS_CACHE=/home/phd_li/.cache/huggingface/hub HF_HUB_OFFLINE=1 python -m llava.serve.cli_roadllm --conv-mode qwen_3 --model-base /home/phd_li/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c --model-path /home/phd_li/git_repo/RoadLLM/checkpoints/projectors/roadllm-llava-openai_clip-vit-large-patch14-336-Qwen_Qwen3-4B-mlp2x_gelu-pretrain-full-4gpus-5epoches --image-file ${IMAGE_NAME}"

bsub -Is -q gpu -gpu "num=1:j_exclusive=yes:gmem=40G" -R "select[type==X64LIN]" eval "${command}"