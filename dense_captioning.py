from transformers import (
    AutoModelForCausalLM, 
    AutoProcessor, 
    GenerationConfig, 
    HfArgumentParser
)
from PIL import Image
import os
from glob import glob
import random
import json
import time
from pathlib import Path

from dataclasses import dataclass, field
from tqdm import tqdm

## For env variable
HF_HUB_OFFLINE = os.getenv("HF_HUB_OFFLINE")
HF_HOME = os.getenv("HF_HOME")
print(f"HF_HUB_OFFLINE: {HF_HUB_OFFLINE}")
print(f"HF_HOME: {HF_HOME}")

@dataclass
class ScriptArguments:
    dataset_name: str = field(
        metadata={"help": "The name of the dataset (key in the config file)"}
    )
    path_config: str = field(
        default="./configs/dataset_paths.json",
        metadata={"help": "Path to the JSON file containing dataset paths"}
    )
    output_root: str = field(
        default="/home/phd_li/dataset/roadllm",
        metadata={"help": "The root directory to save output JSONs"}
    )
    image_count: int = field(
        default=-1,
        metadata={"help": "Number of images to randomly sample. (-1 means all images)"}
    )
    max_new_tokens: int = field(
        default=300,
        metadata={"help": "Maximum new tokens for generation"}
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed for reproducibility"}
    )
    
    path_config_data: dict = field(init=False, repr=False)
    
    def __post_init__(self):
        if not os.path.exists(self.path_config):
            raise FileNotFoundError(
                f"ERROR: The config file '{self.path_config}' was not found. "
                f"Please create it or provide a different path using --path_config."
            )
            
        try:
            with open(self.path_config, 'r') as f:
                self.path_config_data = json.load(f)
            
            for k, v in self.path_config_data.items():
                if not os.path.exists(v):
                    raise FileNotFoundError(
                        f"ERROR: The path for dataset '{k}' specified in '{self.path_config}' does not exist: {v}"
                    )
                    
            if self.dataset_name not in list(self.path_config_data.keys()):
                 raise ValueError(
                     f"ERROR: Dataset '{self.dataset_name}' is not defined in {self.path_config}. "
                     f"Available datasets are: {list(self.path_config_data.keys())}"
                 )
        except json.JSONDecodeError:
            raise ValueError(f"ERROR: '{self.path_config}' is not a valid JSON file.")
        

def main():
    parser = HfArgumentParser((ScriptArguments,))
    # returns a tuple, we only need the first item
    args = parser.parse_args_into_dataclasses()[0]
    
    # read the arguments
    dataset = args.dataset_name
    max_new_tokens = args.max_new_tokens
    image_count = args.image_count
    seed = args.seed
    
    
    # load the processor
    processor = AutoProcessor.from_pretrained(
        'allenai/Molmo-7B-D-0924',
        trust_remote_code=True,
        torch_dtype='auto',
        device_map='auto',
    )

    # load the model
    model = AutoModelForCausalLM.from_pretrained(
        'allenai/Molmo-7B-D-0924',
        trust_remote_code=True,
        torch_dtype='auto',
        device_map='auto',
    )

    
    print(f"Processing data from {dataset}")

    base_path = args.path_config_data[dataset]

    output_dir = os.path.join(args.output_root, dataset)

    os.makedirs(output_dir, exist_ok=True)

    random.seed(seed)
    
    all_image_list = sorted(glob(os.path.join(base_path, '*.jpg')))
    
    if image_count == -1:
        image_list = all_image_list
    else:
        print(f"Randomly sample {image_count} / {len(image_list)} from {dataset}")
        actual_count = min(image_count, len(all_image_list))
        image_list = random.sample(all_image_list, actual_count)

    total_start_time = time.time()

    pbar = tqdm(image_list, desc="Generating Captions")
    
    for image_path in pbar:
        iter_start_time = time.time()
        image = Image.open(image_path)
        
        # process the image and text
        prompt = "Describe this image."
        inputs = processor.process(
            images=[image],
            text=prompt,
        )
        
        # move inputs to the correct device and make a batch of size 1
        inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}
        gen_start_time = time.time()
        # generate output; maximum 200 new tokens; stop generation when <|endoftext|> is generated
        # with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
        output = model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=max_new_tokens, stop_strings="<|endoftext|>"),
            tokenizer=processor.tokenizer
            )
        gen_end_time = time.time()
        # only get generated tokens; decode them to text
        generated_tokens = output[0, inputs['input_ids'].size(1):]
        generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        num_new_tokens = len(generated_tokens)
        gen_duration = gen_end_time - gen_start_time
        total_iter_duration = time.time() - iter_start_time
        
        tok_per_sec = num_new_tokens / gen_duration if gen_duration > 0 else 0
        
        # Update TQDM with metrics
        pbar.set_postfix({
            'toks/sec': f'{tok_per_sec:.1f}',
            'secs/img': f'{total_iter_duration:.2f}'
        })
        
        # print the generated text
        # print("*" * 30)
        # print(generated_text)
        # print()
        
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

    elapsed_time = time.time() - total_start_time
    final_count = len(image_list)
    print(f"Total time elapsed: {elapsed_time:.2f} seconds for {final_count} images. (Avg. {elapsed_time / final_count:.2f} sec / image)")
    


if __name__ == "__main__":
    main()