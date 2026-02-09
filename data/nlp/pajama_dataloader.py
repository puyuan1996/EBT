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

def robust_load_dataset(data_path, split="train", force_reload=False):
    """
    鲁棒地加载数据集，支持智能缓存管理

    修复说明:
    - 添加文件列表指纹检测
    - 智能缓存管理：文件列表变化时自动清除缓存
    - 详细日志输出

    Args:
        data_path: 数据集路径
        split: 分割名称
        force_reload: 是否强制重新加载（忽略缓存）
    """
    print(f"\n{'='*80}")
    print(f"[Data Loader] 正在加载数据集")
    print(f"{'='*80}")
    print(f"路径: {data_path}")
    print(f"强制重载: {force_reload}")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据路径不存在: {data_path}")

    # --- 策略 A: 优先尝试加载 HF Arrow 格式 ---
    if not force_reload:
        if os.path.exists(os.path.join(data_path, "dataset_info.json")) or \
           os.path.exists(os.path.join(data_path, "dataset_dict.json")):
            print("[Data Loader] 检测到 HF Dataset 格式，使用 load_from_disk...")
            try:
                ds = load_from_disk(data_path)
                print(f"✅ 成功加载，样本数: {len(ds):,}")
                if isinstance(ds, DatasetDict):
                    return ds[split] if split in ds else ds[list(ds.keys())[0]]
                return ds
            except Exception as e:
                print(f"[Warning] load_from_disk 失败: {e}")

    # --- 策略 B: 加载原始 JSON/GZ 文件 ---
    print("[Data Loader] 正在搜索原始 .json.gz / .json 文件...")

    # 1. 获取所有文件
    all_files = glob.glob(os.path.join(data_path, "**", "*.json.gz"), recursive=True)
    if not all_files:
        all_files = glob.glob(os.path.join(data_path, "**", "*.json"), recursive=True)

    if not all_files:
        raise FileNotFoundError(f"目录 {data_path} 下未找到任何数据文件。")

    print(f"[Data Loader] 初步找到 {len(all_files):,} 个文件")

    # 2. 智能过滤：RedPajama V2 专用逻辑
    doc_files = [f for f in all_files if "documents" in f]

    if len(doc_files) > 0:
        print(f"[Data Loader] 识别到 RedPajama V2 结构")
        print(f"   - documents: {len(doc_files):,} 个文件")
        print(f"   - quality_signals: {len([f for f in all_files if 'quality_signals' in f]):,} 个文件")
        target_files = doc_files
    else:
        print("[Data Loader] 未检测到标准 'documents' 目录，执行排除法...")
        target_files = [
            f for f in all_files
            if "quality_signals" not in f
            and "metadata" not in f
            and "stats" not in f
        ]
        print(f"[Data Loader] 排除元数据文件后，剩余 {len(target_files):,} 个文件")

    if not target_files:
        raise ValueError("过滤后没有剩余文件！")

    # 3. 【关键修复】缓存管理：基于文件列表哈希
    sorted_files = sorted(target_files)
    files_signature = hashlib.md5(
        str(sorted_files).encode()
    ).hexdigest()[:12]

    cache_dir = os.path.join(data_path, ".cache")
    signature_file = os.path.join(cache_dir, "files_signature.txt")

    # 检查缓存是否过期
    cache_valid = False
    if os.path.exists(signature_file) and not force_reload:
        try:
            with open(signature_file, 'r') as f:
                cached_sig = f.read().strip()
            cache_valid = (cached_sig == files_signature)
            print(f"\n[Cache] 签名对比: {'匹配' if cache_valid else '不匹配'}")
            print(f"   缓存: {cached_sig}")
            print(f"   当前: {files_signature}")
        except:
            pass

    # 【关键修复】如果缓存过期，清除旧缓存
    if (not cache_valid or force_reload) and os.path.exists(cache_dir):
        print(f"[Cache] 清除旧缓存...")
        try:
            shutil.rmtree(cache_dir)
            print(f"[Cache] ✅ 缓存已清除")
        except Exception as e:
            print(f"[Cache] ⚠️  清除失败: {e}")

    # 4. 使用 load_dataset 加载数据
    print(f"\n[Data Loader] 开始加载 {len(target_files):,} 个文件...")
    print(f"[Data Loader] 估计样本数: ~{len(target_files) * 27000:,}")

    try:
        os.makedirs(cache_dir, exist_ok=True)

        ds = load_dataset(
            "json",
            data_files=sorted_files,  # 使用排序后的列表
            split="train",
            cache_dir=cache_dir,
            download_mode='force_redownload' if (not cache_valid or force_reload) else None,
            # num_proc=1  # 如果遇到死锁，取消注释此行
        )

        # 保存文件签名
        with open(signature_file, 'w') as f:
            f.write(files_signature)

    except Exception as e:
        print(f"[Error] 标准 JSON 加载失败: {e}")
        print("[Data Loader] 尝试使用【生成器模式】...")
        return load_dataset_via_generator(sorted_files)

    # 5. 统一列名
    print(f"\n[Data Loader] 加载成功！样本数: {len(ds):,}")

    if "raw_content" in ds.column_names and "text" not in ds.column_names:
        print("[Data Loader] 重命名列: raw_content -> text")
        ds = ds.rename_column("raw_content", "text")

    print(f"{'='*80}\n")
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
