import argparse
import torch
import json
import os
import requests
import copy
from io import BytesIO
from PIL import Image

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path

def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image

def main(args):
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    # Load model
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name, 
        args.load_8bit, args.load_4bit, torch_dtype=args.dtype
    )

    # Determine conversation mode
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print(f"[WARNING] using provided conv-mode: {args.conv_mode}")
    else:
        args.conv_mode = conv_mode

    # Setup conversation
    conv = conv_templates[args.conv_mode].copy()
    
    # Process Image
    image = load_image(args.image_file)
    if args.dtype == "float16":
        image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"].half().cuda()
    elif args.dtype == "bfloat16":
        image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"].bfloat16().cuda()
    else:
        # Default fallback
        image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"].cuda()

    # Handle Special Tokens
    tokenizer = copy.deepcopy(tokenizer)
    if "<image>" not in tokenizer.get_vocab():
        tokenizer.add_tokens(["<image>"], special_tokens=True)
    if "<|im_start|>" not in tokenizer.get_vocab():
        tokenizer.add_tokens(["<|im_start|>", "<|im_end|>"], special_tokens=True)

    # Set Chat Template
    tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"

    # Prepare Prompt
    inp = args.prompt
    if image is not None:
        content = DEFAULT_IMAGE_TOKEN + "\n" + inp
    else:
        content = inp

    history = [{"role": "system", "content": "You are a helpful assistant."}]
    history.append({"role": "user", "content": content})

    # Tokenize
    input_ids = tokenizer.apply_chat_template(history, add_generation_prompt=True, return_tensors="pt")
    image_token_id = tokenizer.convert_tokens_to_ids("<image>")
    input_ids[input_ids == image_token_id] = IMAGE_TOKEN_INDEX
    input_ids = input_ids.to(device=model.device)
    attention_mask = torch.ones_like(input_ids, device=input_ids.device).bool()

    # Generate
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            attention_mask=attention_mask,
            do_sample=True, # You can change to False for deterministic output
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id 
        )

    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=False).strip()
    print(f"Model: {model_name}\nOutput: {outputs}\n")

    # Save to JSON
    result_entry = {
        "model_size": args.model_alias, # passed from bash script for easy ID
        "model_path": args.model_path,
        "prompt": args.prompt,
        "image_file": args.image_file,
        "response": outputs
    }

    # If output file exists, append to list; otherwise create new list
    if os.path.exists(args.output_file):
        with open(args.output_file, 'r') as f:
            try:
                data = json.load(f)
                if not isinstance(data, list): data = [data]
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    data.append(result_entry)

    with open(args.output_file, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Saved result to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--model-alias", type=str, default="unknown", help="Alias like 0.6B, 8B etc")
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="Describe the image")
    parser.add_argument("--output-file", type=str, default="results.json")
    parser.add_argument("--conv-mode", type=str, default="qwen_3")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    args = parser.parse_args()
    main(args)