from datasets import load_dataset, load_from_disk
from torch.utils.data import Dataset
from functools import partial
from datasets import Dataset as hf_Dataset
import os
import json


class GSM8KDataset(Dataset):
    def __init__(self, hparams, split):
        self.hparams = hparams

        # Path to offline dataset (update this path after downloading the dataset)
        local_dataset_path = "/mnt/shared-storage-user/puyuan/code/EBT/data/gsm8k_offline"

        # Try to load from local path first, fallback to online download
        if os.path.exists(local_dataset_path):
            print(f"Loading GSM8K dataset from local path: {local_dataset_path}")
            dataset = load_from_disk(local_dataset_path)
            self.dataset = dataset[split]
        else:
            print(f"Local dataset not found at {local_dataset_path}, downloading from HuggingFace...")
            hf_token = os.getenv('HF_TOKEN')
            hf_home = os.getenv('HF_HOME')
            dataset_dir = self.hparams.dataset_dir if self.hparams.dataset_dir != "" else hf_home
            self.dataset = load_dataset("openai/gsm8k", "main", cache_dir=dataset_dir, token=hf_token, trust_remote_code=True)[split]
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if self.hparams.execution_mode == "inference":
            return f"[[Question]]: {self.dataset[idx]['question']}\n[[Answer]]: ", self.dataset[idx]['answer']
        elif self.hparams.execution_mode == "pretrain":
            return f"Question: {self.dataset[idx]['question']}\nAnswer: {self.dataset[idx]['answer']}"
        elif self.hparams.execution_mode == "finetune":
            return f"[[Question]]: {self.dataset[idx]['question']}\n[[Answer]]: {self.dataset[idx]['answer']}"
        else:
            raise ValueError(f"Execution mode not supported. Please add support for mode {self.hparams.execution_mode}")