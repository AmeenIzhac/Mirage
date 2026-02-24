import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import os
import json
import logging
from tqdm import tqdm

from qwen_vl_utils import process_vision_info

from utils import *
from task import *

seed_everything(seed=42)
args=get_args()

# Override the data path for comfort test
if args.data_path == 'PathToJsonlData' or args.data_path is None:
    args.data_path = "./data/comfort/comfort_car_sample.jsonl"

logging.basicConfig(
    level=logging.INFO,  # Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
    datefmt='%Y-%m-%d %H:%M:%S',  # Date format
    handlers=[
        logging.FileHandler(args.log_file, mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ],
)

logging.info('=='*20)
logging.info(args)
logging.info('=='*20)
    
# Load the model and processor
cache_dir = args.cache_dir
os.environ['HF_HOME'] = cache_dir

processor = AutoProcessor.from_pretrained(args.load_model_path, trust_remote_code=True, cache_dir=cache_dir)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.load_model_path, device_map="auto", torch_dtype=torch.bfloat16)

processor.tokenizer.add_tokens("<|latent_pad|>", special_tokens=True)
processor.tokenizer.add_tokens("<|latent_start|>", special_tokens=True)
processor.tokenizer.add_tokens("<|latent_end|>", special_tokens=True)

model.eval()

with open(args.data_path, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

correct = 0
for i, sample in tqdm(enumerate(data)):
    
    # Fix image path (dataset uses ./data/comfort/comfort/ but it's actually ./data/comfort/images/)
    sample["image_input"] = sample["image_input"].replace("/comfort/", "/images/")
    
    preprocess_function = single_input_image_test_preprocess_function
    conversations = preprocess_function(sample)

    texts = [processor.apply_chat_template(conversations, tokenize=False)]
    texts = [place_input_image(text, sep_token=None) for text in texts]
    image_inputs, _ = process_vision_info(conversations)

    inputs = processor(text=[t+'<|im_start|>assistant' for t in texts], images=image_inputs, return_tensors="pt", padding=True)
    inputs = inputs.to(model.device)
    
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=10, # Minimal tokens for MC answer
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            tokenizer=processor.tokenizer,
        )
    
    decoded_output = processor.tokenizer.decode(output_ids[0], skip_special_tokens=False)
    answer = decoded_output.split('<|im_start|>assistant')[-1].strip()

    target = sample.get("correct_answer", "").strip()
    
    if answer == target: 
        correct += 1

    if (i+1) % 20 == 0:
        logging.info(f"[{i+1}] Accuracy: {correct}/{i+1} ({correct/(i+1):.3f})")

logging.info(f"[Final] Accuracy: {correct}/{len(data)} ({correct/len(data):.3f})")
