import argparse
import os
import os.path as osp



def parse_args(dataset_name: str = None, task: str = None, topic: str = None, text_input_length: int = None,
               model_name: str = None, split: str = None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--hf_cache_dir", type=str, default="/workingdir/yjin328/cache")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--model_name", type=str, required=True)

    args = parser.parse_args()

    return args
