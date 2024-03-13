import os
import json

import torch
from datasets import load_dataset
from peft import PeftModel, LoraConfig
from transformers import LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig

from arguments import parse_args
from utils import print_colored

os.environ['HF_DATASETS_CACHE'] = '/workingdir/yjin328/cache'


def load_model(model_name, quantization):
    quantization_config = BitsAndBytesConfig(load_in_8bit=True, load_in_4bit=False)
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        return_dict=True,
        load_in_8bit=quantization,
        quantization_config=quantization_config,
        device_map="auto",
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
        offload_folder="/workingdir/yjin328/offload",
        offload_state_dict=True,
    )
    return model

# Function to load the PeftModel for performance optimization
def load_peft_model(model):
    peft_model = PeftModel.from_pretrained(model, peft_model)
    return peft_model

def generate_answer_llama(model_name: str, dataset_name: str, adapter_path: str=None, load_in_8bit: bool = False, use_peft_model:bool=False, max_new_tokens: int=128, hf_cache_dir: str=None):
    model = load_model(model_name, load_in_8bit)


    if adapter_path is not None:
        print(f"Loading adapter from {adapter_path}")
        # load a pre-trained model.
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
        finetuned_config = LoraConfig.from_pretrained(pretrained_model_name_or_path=adapter_path)

    else:
        finetuned_config = None

    if use_peft_model:
        model = load_peft_model(model)

    model.eval()

    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens(
        {
            "pad_token": "<PAD>",
        }
    )

    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model.resize_token_embeddings(model.config.vocab_size + 1)


    # for split in ["low", "medium", "high", "veryhigh"]:
    for split in ["veryhigh"]:

        # dataset_name = "Ahren09/RealToxicityPrompts_val_100"
        dataset = load_dataset(dataset_name, split=split, cache_dir=hf_cache_dir)
        all_prompts = dataset['prompt']
        # all_prompts_encoded = tokenizer.encode(all_prompts, return_tensors=)
        all_prompts_encoded = tokenizer(all_prompts, padding='max_length', truncation=True, max_length=args.max_new_tokens,
                          return_tensors="pt")

        all_output_texts = []
        path = f"outputs/answers_{model_name.split('/')[-1]}_{dataset_name.split('/')[-1].replace('_', '-')}_{split}.json"
        print(f"Saving to {path}")

        for idx_start in range(0, len(dataset), args.batch_size):
            # batch = dataset.select(idx_start, min(idx_start + args.batch_size, len(dataset)))
            batch = all_prompts_encoded[idx_start: idx_start + args.batch_size]

            batch = {k: v.to(model.device) for k, v in batch.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **batch,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    top_p=0.9,
                    temperature=1.0,
                    min_length=8,
                    use_cache=True,
                    top_k=5,
                    num_beams=5,
                    repetition_penalty=1.5,
                    length_penalty=1.2,
                )
                for i in range(outputs.shape[0]):
                    all_output_texts += [tokenizer.decode(outputs[i], skip_special_tokens=True)]
                    print_colored('=' * 30 + f"\n{idx_start} - Question:", 'blue')
                    print_colored(all_prompts[idx_start + i], "blue")
                    print_colored(f"{idx_start} - Answer", "yellow")
                    print_colored(all_output_texts[-1], "yellow")


        json.dump(all_output_texts, open(path, 'w'), indent=2)

if __name__ == "__main__":
    args = parse_args()

    generate_answer_llama(args.model_name, args.dataset_name, args.adapter_path, load_in_8bit=True, max_new_tokens=args.max_new_tokens, hf_cache_dir=args.hf_cache_dir)