import logging
from dataclasses import dataclass, field
import os
import random
import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments
from trl.commands.cli_utils import  TrlParser
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
        set_seed,
    EarlyStoppingCallback,

)
from trl import setup_chat_format
from peft import LoraConfig
from tqdm import tqdm


from trl import (
   SFTTrainer)



# LLAMA_3_CHAT_TEMPLATE="{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"

# Anthropic/Vicuna like template without the need for special tokens
LLAMA_3_CHAT_TEMPLATE = (
    "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
            "{{ message['content'] }}"
        "{% elif message['role'] == 'user' %}"
            "{{ '\n\nHuman: ' + message['content'] +  eos_token }}"
        "{% elif message['role'] == 'assistant' %}"
            "{{ '\n\nAssistant: '  + message['content'] +  eos_token  }}"
        "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '\n\nAssistant: ' }}"
    "{% endif %}"
)



@dataclass
class ScriptArguments:
    dataset_path: str = field(
        default=None,
        metadata={
            "help": "Path to the dataset"
        },
    )
    model_id: str = field(
        default=None, metadata={"help": "Model ID to use for SFT training"}
    )
    max_seq_length: int = field(
        default=512, metadata={"help": "The maximum sequence length for SFT Trainer"}
    )
    dataset_type: str = field(
        default=None,
        metadata={
            "help": "Type of the dataset"
        },
    )
    checkpoint_dir: str = field(
        default=None,
        metadata={
            "help": "Path to the checkpoint"
        },
    )

    

def inference_function(script_args, training_args):

    tokenizer = AutoTokenizer.from_pretrained(f"./training_result/{script_args.dataset_type}/{script_args.checkpoint_dir}")

    model = AutoModelForCausalLM.from_pretrained(
        f"./training_result/{script_args.dataset_type}/{script_args.checkpoint_dir}",
        torch_dtype=torch.float32,
        device_map="auto"
    )


    test_dataset = load_dataset(
        "json",
        data_files=os.path.join(script_args.dataset_path, f"{script_args.dataset_type}_test_dataset.json"),
        split="train",
    )
    print('start generating')
    all_response = []
    for data in tqdm(test_dataset):
        messages = data['messages'][:2]

        input_ids = tokenizer.apply_chat_template(messages,add_generation_prompt=True,return_tensors="pt").to(model.device)
        outputs = model.generate(
            input_ids,
            max_new_tokens=512,
            eos_token_id= tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.01,
            top_p=0.9,
        )
        response = outputs[0][input_ids.shape[-1]:]
        all_response.append(tokenizer.decode(response,skip_special_tokens=True))
    
    with open(f'./inference_result/{script_args.dataset_type}_response.json', 'w', encoding='utf-8') as f:
        json.dump(all_response, f)

if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, TrainingArguments))
    script_args, training_args = parser.parse_args_and_config()    
    
    # set use reentrant to False
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}
    # set seed
    set_seed(training_args.seed)

    # inference the result
    inference_function(script_args, training_args)