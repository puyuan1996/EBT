from transformers import AutoTokenizer
from datasets import load_dataset, get_dataset_config_names, load_from_disk
import torch
from torch.utils.data import Dataset
from functools import partial
from datasets import Dataset as hf_Dataset
import os
import json

import os
import glob
from datasets import load_from_disk, load_dataset, DatasetDict

import os
import glob
import json
import gzip
import hashlib
import shutil
from datasets import load_from_disk, load_dataset, DatasetDict, Dataset

def robust_load_dataset(data_path, split="train"):
    print(f"[Data Loader] 正在扫描路径: {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据路径不存在: {data_path}")

    # --- 策略 A: 优先尝试加载 HF Arrow 格式 (速度最快) ---
    if os.path.exists(os.path.join(data_path, "dataset_info.json")) or \
       os.path.exists(os.path.join(data_path, "dataset_dict.json")):
        print("[Data Loader] 检测到 HF Dataset 格式，使用 load_from_disk...")
        try:
            ds = load_from_disk(data_path)
            if isinstance(ds, DatasetDict):
                return ds[split] if split in ds else ds[list(ds.keys())[0]]
            return ds
        except Exception as e:
            print(f"[Warning] load_from_disk 失败: {e}")

    # --- 策略 B: 加载原始 JSON/GZ 文件 (增加智能过滤) ---
    print("[Data Loader] 正在搜索原始 .json.gz / .json 文件...")
    
    # 1. 获取所有文件
    all_files = glob.glob(os.path.join(data_path, "**", "*.json.gz"), recursive=True)
    if not all_files:
        all_files = glob.glob(os.path.join(data_path, "**", "*.json"), recursive=True)
    
    if not all_files:
        raise FileNotFoundError(f"目录 {data_path} 下未找到任何数据文件。")

    print(f"[Data Loader] 初步找到 {len(all_files)} 个文件。正在进行智能过滤...")

    # 2. 智能过滤：RedPajama V2 专用逻辑
    # 逻辑：如果路径中包含 'documents'，通常是正文；如果包含 'quality_signals' 或 'metadata'，通常是纯元数据，需要剔除。
    
    # 优先寻找包含 'documents' 路径的文件
    doc_files = [f for f in all_files if "documents" in f]
    
    # 如果找到了 documents 文件夹下的内容，就只用这些
    if len(doc_files) > 0:
        print(f"[Data Loader] 识别到 RedPajama V2 结构，仅保留 'documents' 目录下的 {len(doc_files)} 个文本文件。")
        target_files = doc_files
    else:
        # 如果没找到 documents 目录，则使用所有文件，但排除明确的非文本目录
        print("[Data Loader] 未检测到标准 'documents' 目录，执行排除法...")
        target_files = [
            f for f in all_files 
            if "quality_signals" not in f 
            and "metadata" not in f 
            and "stats" not in f
        ]
        print(f"[Data Loader] 排除元数据文件后，剩余 {len(target_files)} 个文件。")

    if not target_files:
        raise ValueError("过滤后没有剩余文件！请检查你的数据目录是否只包含 quality_signals 而没有 documents？")

    # 3. 使用 load_dataset 加载清洗后的文件列表
    print("[Data Loader] 开始加载数据...")
    try:
        ds = load_dataset(
            "json", 
            data_files=target_files, 
            split="train",
            cache_dir=os.path.join(data_path, ".cache"),
            # num_proc=1 # 如果遇到死锁，取消注释此行
        )
    except Exception as e:
        print(f"[Error] 标准 JSON 加载失败: {e}")
        print("[Data Loader] 尝试使用【生成器模式】作为最后手段（速度较慢但最鲁棒）...")
        return load_dataset_via_generator(target_files)

    # 4. 统一列名 (RedPajama V2 raw_content -> text)
    if "raw_content" in ds.column_names and "text" not in ds.column_names:
        print("[Data Loader] 重命名列: raw_content -> text")
        ds = ds.rename_column("raw_content", "text")
        
    # 5. 移除不必要的列以节省内存 (可选，防止后续处理报错)
    # 保留核心列，防止 metadata 里的奇怪结构导致后面报错
    keep_cols = {"text", "id", "meta", "source"}
    cols_to_remove = [c for c in ds.column_names if c not in keep_cols]
    if cols_to_remove:
        # 这里不真正删除，只是打印日志，防止误删有用信息。
        # 如果后续训练代码报错有多余列，可以在这里 ds.remove_columns(cols_to_remove)
        pass

    return ds

def load_dataset_via_generator(file_list):
    """
    Plan B: 如果 JSON 结构极其混乱（有的有 id，有的没有），
    使用 Python 生成器强制只读取 text 字段。这绝对不会报错，但速度稍慢。
    """
    def gen():
        for file_path in file_list:
            try:
                # 自动判断是否需要 gzip
                open_func = gzip.open if file_path.endswith(".gz") else open
                with open_func(file_path, "rt", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip(): continue
                        try:
                            item = json.loads(line)
                            # 暴力提取文本，忽略其他所有不一致的字段
                            text = item.get("raw_content", item.get("text", item.get("content", "")))
                            if text:
                                yield {"text": text}
                        except:
                            continue
            except Exception as e:
                print(f"无法读取文件 {file_path}: {e}")
                continue

    return Dataset.from_generator(gen)

class RedPajamaDataset(Dataset):
    def __init__(self, hparams): # dont use tokenizer is in collator
        if hparams.execution_mode != "pretrain":
            raise ValueError("RedPajama is a pretrain dataset, no other execution modes supported.")
            
        #NOTE there is only 1 split (train) so every other split does the same here
        self.max_length = hparams.context_length+1
        hf_home = os.getenv('HF_HOME')

        # Get dataset_dir with proper fallback logic
        dataset_dir = getattr(hparams, 'dataset_dir', None)
        if dataset_dir is None or dataset_dir == "":
            dataset_dir = hf_home
        if dataset_dir is None:
            # Final fallback to default HuggingFace cache directory
            dataset_dir = os.path.expanduser("~/.cache/huggingface")
        self.tokenizer = AutoTokenizer.from_pretrained(hparams.tokenizer, clean_up_tokenization_spaces = False)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id # just for reference the tokenizer is fast

        if hparams.pretokenize_dataset:
            save_path = os.path.join(dataset_dir, hparams.dataset_name + '_preprocessed', hparams.tokenizer.replace('/', '_'), "max_length_" + str(self.max_length))
            print("pretokenized dataset save_path", save_path)

            if os.path.exists(save_path): # load dataset it exists
                print(f"loading {hparams.dataset_name} dataset")
                self.dataset = load_from_disk(save_path)
            else: # need to create dataset
                print(f"no pre-tokenized {hparams.dataset_name} dataset with correct settings, loading and saving")
                # self.dataset = load_dataset("togethercomputer/RedPajama-Data-V2", "sample-100B", split = "train", cache_dir=dataset_dir, trust_remote_code=True, keep_in_memory = False)

                # TODO
                self.dataset = robust_load_dataset("/mnt/shared-storage-user/puyuan/code/EBT/dataset/RedPajama-Data-V2-Local")

                num_proc = hparams.num_workers * hparams.num_gpus
                print("num_proc using for dataset map", num_proc) # found that if have 192 cpus then cannot use 96 (it freezes), so 48 was good. make sure to test this with your own hardware and adjust num workers accordingly
                # NOTE this code may freeze and takes a very long time to run, make sure to test what values for num_proc and num_workers are best
                self.dataset = self.dataset.map(self.tokenization, num_proc = num_proc) # batched=True, batch_size=hparams.batch_size_per_device,
                print("done preprocessing dataset")
                self.dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
                print("done formatting dataset")
                self.dataset.save_to_disk(save_path)
        else:
            self.dataset = load_dataset("togethercomputer/RedPajama-Data-V2", "sample-100B", split = "train", cache_dir=dataset_dir, trust_remote_code=True, keep_in_memory = False)

        self.hparams = hparams

    def tokenization(self, example):
        # 兼容不同的字段名：text（重命名后）或 raw_content（原始）
        text_content = example.get('text', example.get('raw_content', ''))
        return self.tokenizer(text_content, padding=True, truncation=True, max_length=self.max_length)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        if self.hparams.pretokenize_dataset:
            return self.dataset[idx]
        else:
            return self.dataset[idx]['raw_content']
