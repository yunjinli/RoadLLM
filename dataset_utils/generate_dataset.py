import argparse
import json
import os
import random
from glob import glob
from pathlib import Path
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Generate LLaVA-compatible pretraining dataset JSON.")
    
    parser.add_argument(
        "--config_path", 
        type=str, 
        required=True,
        default="./configs/dataset_paths.json",
        help="Path to the dataset_paths.json config file."
    )
    parser.add_argument(
        "--image_base_path", 
        type=str, 
        required=True,
        help="Root directory for images. Used to calculate relative paths."
    )
    parser.add_argument(
        "--caption_base_path", 
        type=str, 
        required=True,
        help="Root directory containing subfolders of caption JSONs."
    )
    parser.add_argument(
        "--instructions_path", 
        type=str, 
        required=True,
        default="./configs/detailed_image_description_instructions.json",
        help="Path to the JSON file containing the list of instructions."
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        default="./configs/pretrained.json",
        help="Path where the output JSON will be saved."
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for reproducibility."
    )

    return parser.parse_args()

def main():
    args = parse_args()
    
    random.seed(args.seed)

    print(f"Loading instructions from {args.instructions_path}...")
    try:
        with open(args.instructions_path, 'r') as f:
            detailed_img_desc_inst = json.load(f)
            if not isinstance(detailed_img_desc_inst, list):
                raise ValueError("Instruction file must contain a list of strings.")
    except Exception as e:
        print(f"Error loading instructions: {e}")
        return

    print(f"Loading config from {args.config_path}...")
    try:
        with open(args.config_path, 'r') as file:
            config_dataset_base_dirs = json.load(file)
    except FileNotFoundError:
        print(f"Error: Config file not found at {args.config_path}")
        return

    search_pattern = os.path.join(args.caption_base_path, "*", "*.json")
    caption_paths = sorted(glob(search_pattern))
    
    if not caption_paths:
        print(f"Warning: No JSON files found matching pattern: {search_pattern}")
        return

    print(f"Found {len(caption_paths)} caption files. Processing...")

    annotation = []

    for caption_path in tqdm(caption_paths, desc="Processing Captions"):
        try:
            dataset_name = Path(caption_path).parent.name
            
            # Check if dataset name exists in config
            if dataset_name not in config_dataset_base_dirs:
                # Optional: Skip or Log error if dataset not in config
                continue

            with open(caption_path, 'r') as file:
                caption = json.load(file)
                
                # Construct absolute path based on config mapping
                abs_image_path = os.path.join(config_dataset_base_dirs[dataset_name], caption['image'])
                
                # Convert to path relative to the image_base_path provided in args
                # This ensures the output JSON points to images relative to your project root
                try:
                    rel_path = Path(abs_image_path).relative_to(os.path.abspath(args.image_base_path))
                    caption['image'] = str(rel_path)
                except ValueError:
                    # Fallback or error if image is not inside image_base_path
                    print(f"Warning: {abs_image_path} is not inside {os.path.abspath(args.image_base_path)}")
                    caption['image'] = abs_image_path 

                # Inject random instruction
                if 'conversations' in caption and len(caption['conversations']) > 0:
                    caption['conversations'][0]['value'] = "<image>\n" + random.choice(detailed_img_desc_inst)
                
                annotation.append(caption)

        except Exception as e:
            print(f"Error processing file {caption_path}: {e}")
            continue

    print(f"Saving {len(annotation)} annotations to {args.output_path}...")
    with open(args.output_path, 'w') as outfile:
        json.dump(annotation, outfile, indent=4)
    
    print("Done.")

if __name__ == "__main__":
    main()