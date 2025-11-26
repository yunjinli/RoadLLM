from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
import os
from glob import glob
import random
import torch
import json
import time
import sys
from pathlib import Path
## For env variable
HF_HUB_OFFLINE = os.getenv("HF_HUB_OFFLINE")
HF_HOME = os.getenv("HF_HOME")
print(f"HF_HUB_OFFLINE: {HF_HUB_OFFLINE}")
print(f"HF_HOME: {HF_HOME}")

# load the processor
processor = AutoProcessor.from_pretrained(
    'allenai/Molmo-7B-D-0924',
    trust_remote_code=True,
    torch_dtype='auto',
    device_map='auto',
    # cache_dir=CACHE_DIR,
)

# load the model
model = AutoModelForCausalLM.from_pretrained(
    'allenai/Molmo-7B-D-0924',
    trust_remote_code=True,
    torch_dtype='auto',
    device_map='auto',
    # cache_dir=CACHE_DIR,
)

dataset = sys.argv[1]
print(f"Processing data from {dataset}")


BASE_PATH = {
    'r2s100k': "/home/phd_li/dataset/r2s100k/train",
    'bdd100k': "/home/phd_li/dataset/bdd100k/images/10k/train",
    'rdd': "/home/phd_li/dataset/RDD2022/train2017",
}

assert dataset in list(BASE_PATH.keys())

base_path = BASE_PATH[dataset]

# output_dir = f"/home/phd_li/git_repo/RoadLLM/Molmo-7B-D-0924/{dataset}"
output_dir = f"/home/phd_li/dataset/roadllm/{dataset}"

# output_image_path = os.path.join(output_dir, 'images')

os.makedirs(output_dir, exist_ok=True)

image_count = 10

# size = (320, 240)

max_new_tokens = 300

image_list = sorted(glob(os.path.join(base_path, '*.jpg')))

print(f"Randomly sample {image_count} / {len(image_list)} from {dataset}")
# images = []
seed = 42
random.seed(seed)
# captions = []

start_time = time.time()

for i in range(image_count):
    image_path = random.choice(image_list)
    # image = Image.open(image_path).resize(size)
    image = Image.open(image_path)
    # image.save(os.path.join(output_image_path, os.path.basename(image_path)))
    # images.append(image)
    # process the image and text
    prompt = "Describe this image."
    inputs = processor.process(
        images=[image],
        text=prompt,
    )
    # move inputs to the correct device and make a batch of size 1
    inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

    # generate output; maximum 200 new tokens; stop generation when <|endoftext|> is generated
    # with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
    output = model.generate_from_batch(
        inputs,
        GenerationConfig(max_new_tokens=max_new_tokens, stop_strings="<|endoftext|>"),
        tokenizer=processor.tokenizer
        )

    # only get generated tokens; decode them to text
    generated_tokens = output[0, inputs['input_ids'].size(1):]
    generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    # print the generated text
    print("*" * 30)
    print(generated_text)
    print()
    # captions.append({'image_id': os.path.basename(image_path), 'caption': generated_text})
    caption = {
        "id": Path(image_path).stem,
        "image": os.path.basename(image_path),
        "conversations": [
        {
            "from": "human",
            "value": prompt
        },
        {
            "from": "gpt",
            "value": generated_text
        }
        ]
    }
    json_str = json.dumps(caption, indent=4)
    with open(os.path.join(output_dir, f"{Path(image_path).stem}.json"), "w") as f:
        f.write(json_str)

# End timer
end_time = time.time()

# Calculate and print time elapsed
elapsed_time = end_time - start_time
print(f"Time elapsed: {elapsed_time:.2f} seconds for {image_count} images. (Avg. {elapsed_time / image_count:.2f} sec / image)")



