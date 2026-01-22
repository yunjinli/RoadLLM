import argparse
import torch

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image

import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer
import copy


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def main(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    # tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, torch_dtype="float16")
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name, args.load_8bit, args.load_4bit, torch_dtype=args.dtype)

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print("[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}".format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    # print(args.conv_mode)
    conv = conv_templates[args.conv_mode].copy()
    if "mpt" in model_name.lower():
        roles = ("user", "assistant")
    else:
        roles = conv.roles

    image = load_image(args.image_file)
    if args.dtype == "float16":
        image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"].half().cuda()
    elif args.dtype == "bfloat16":
        image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"].bfloat16().cuda()
    else:
        raise ValueError("Only float16 and bfloat16 are supported for image tensor")
    # image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"].half().cuda()
    # image_tensor = image_processor.preprocess(image, return_tensors="pt")["pixel_values"].bfloat16().cuda()

    tokenizer = copy.deepcopy(tokenizer)
    if "<image>" not in tokenizer.get_vocab():
        tokenizer.add_tokens(["<image>"], special_tokens=True)
    
    # Ensure Qwen Special Tokens exist
    if "<|im_start|>" not in tokenizer.get_vocab():
        tokenizer.add_tokens(["<|im_start|>", "<|im_end|>"], special_tokens=True)
    
    # 3. Set the Chat Template manually (Matches your train.py)
    tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    history = [{"role": "system", "content": "You are a helpful assistant."}]
    # history = [{"role": "system", "content": "You are a helpful assistant. /think"}]
    
    print(f"Tokenizer Vocab Size: {len(tokenizer)}")
    print(f"Model Embedding Size: {model.get_input_embeddings().weight.shape[0]}")
        
    while True:
        try:
            inp = input(f"{roles[0]}: ")
        except EOFError:
            inp = ""
        if not inp:
            print("exit...")
            break

        print(f"{roles[1]}: ", end="")

        if image is not None:
            content = DEFAULT_IMAGE_TOKEN + "\n" + inp
        else:
            content = inp
        history.append({"role": "user", "content": content})
        input_ids = tokenizer.apply_chat_template(history, add_generation_prompt=True, return_tensors="pt")
        # decoded_text = tokenizer.decode(input_ids[0], skip_special_tokens=False)
        # print(decoded_text)
        image_token_id = tokenizer.convert_tokens_to_ids("<image>")
        input_ids[input_ids == image_token_id] = IMAGE_TOKEN_INDEX
        
        input_ids = input_ids.to(device=model.device)
        attention_mask = torch.ones_like(input_ids, device=input_ids.device).bool()

        # print(history)
        print(input_ids.shape)
        
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)
        # Note: stopping criteria is less strict here for simplicity, relying on EOS token

        print("RoadLLM: ", end="")
        
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                attention_mask=attention_mask,
                do_sample=True,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                streamer=streamer,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id 
            )

        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=False).strip()
        
        # Add assistant response to history
        history.append({"role": "assistant", "content": outputs})
        
        # Clear image after first turn so we don't re-inject it
        if image is not None:
            image = None
            image_tensor = None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)
