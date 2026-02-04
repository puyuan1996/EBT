import os
import requests
import gzip
import shutil
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import time

# ================= 配置区域 =================
# 1. 保存根目录 (请修改为你服务器上的实际路径)
SAVE_ROOT = "/mnt/shared-storage-user/puyuan/code/EBT/dataset/RedPajama-Data-V2-Local"

# 2. 下载模式
# "sample": 只下载每个分类的前 N 个文件 (用于测试)
# "full": 下载清单中的所有文件 (慎用！数据量巨大)
MODE = "sample" 

# pajama 100B样本 一共10000条 大概需要1000g
SAMPLE_COUNT = 500  # 41G 

# 3. 数据配置
# 可选语言: "en", "de", "fr", "es", "it"
LANGUAGES = ["en"] 
# 可选快照: 参考官方列表，例如 "2023-06", "2022-49"
SNAPSHOTS = ["2023-06"] 
# 分区: "head_middle" (高质量/去重后) 或 "tail"
PARTITION = "head_middle"

# 4. 下载组件
# documents: 纯文本数据 (必须)
# quality_signals: 质量评分元数据 (推荐)
# duplicates: 重复文档ID
# minhash: 用于去重的哈希签名
COMPONENTS = ["documents", "quality_signals"] 

# 5. 并发数 (根据你的网络带宽调整)
MAX_WORKERS = 4

# 基础 URL (来自官方文档)
BASE_URL = "https://data.together.xyz/redpajama-data-v2/v1.0.0"
# ===========================================

def download_file(url, local_path):
    """
    下载单个文件，支持断点续传（通过检查文件存在性），带进度条
    """
    if os.path.exists(local_path):
        # 简单的检查：如果文件存在且大小不为0，跳过
        # 生产环境可以加上校验和检查，但 HTTP header 通常不给 MD5
        if os.path.getsize(local_path) > 0:
            print(f"[跳过] 已存在: {local_path}")
            return True
        else:
            os.remove(local_path) # 删除空文件重新下载

    # 确保父目录存在
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    try:
        # 流式下载
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            
            with open(local_path, 'wb') as f, tqdm(
                desc=os.path.basename(local_path),
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
                leave=False
            ) as bar:
                for chunk in r.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    bar.update(size)
        return True
    except Exception as e:
        print(f"\n[错误] 下载失败 {url}: {e}")
        if os.path.exists(local_path):
            os.remove(local_path) # 下载失败清理残余
        return False

def get_listing_urls(lang, snapshot, partition):
    """
    获取官方的文件清单 (Listings)
    """
    listing_tag = f"{lang}-{snapshot}-{partition}"
    listing_url = f"{BASE_URL}/listings/{listing_tag}.txt"
    listing_local_path = os.path.join(SAVE_ROOT, "listings", f"{listing_tag}.txt")
    
    print(f"正在获取清单文件: {listing_tag} ...")
    if download_file(listing_url, listing_local_path):
        with open(listing_local_path, 'r') as f:
            # 读取每一行，去除空白符
            files = [line.strip() for line in f if line.strip()]
        return files
    else:
        print(f"无法获取清单: {listing_url}")
        return []

def main():
    print(f"=== RedPajama-V2 下载器启动 ===")
    print(f"模式: {MODE} | 保存路径: {SAVE_ROOT}")
    
    for lang in LANGUAGES:
        for snapshot in SNAPSHOTS:
            # 1. 获取该配置下的所有文件列表
            file_ids = get_listing_urls(lang, snapshot, PARTITION)
            
            if not file_ids:
                continue

            print(f"\n找到 {len(file_ids)} 个文件 ID (快照: {snapshot}, 语言: {lang})")

            # 如果是采样模式，截取前 N 个
            target_files = file_ids
            if MODE == "sample":
                target_files = file_ids[:SAMPLE_COUNT]
                print(f"采样模式：仅下载前 {len(target_files)} 个文件")

            # 2. 遍历组件进行下载
            for component in COMPONENTS:
                print(f"\n--- 正在处理组件: {component} ---")
                
                # 构建下载任务列表
                tasks = []
                for file_id in target_files:
                    # 根据组件不同，后缀名可能不同
                    # documents -> .json.gz
                    # quality_signals -> .signals.json.gz
                    # duplicates -> .duplicates.parquet
                    # minhash -> .minhash.parquet
                    
                    if component == "documents":
                        suffix = ".json.gz"
                        remote_path = f"documents/{file_id}{suffix}"
                    elif component == "quality_signals":
                        suffix = ".signals.json.gz"
                        remote_path = f"quality_signals/{file_id}{suffix}"
                    elif component == "duplicates":
                        suffix = ".duplicates.parquet"
                        remote_path = f"duplicates/{file_id}{suffix}"
                    elif component == "minhash":
                        suffix = ".minhash.parquet"
                        remote_path = f"minhash/{file_id}{suffix}"
                    else:
                        continue

                    url = f"{BASE_URL}/{remote_path}"
                    local_path = os.path.join(SAVE_ROOT, remote_path)
                    tasks.append((url, local_path))

                # 3. 多线程执行下载
                with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                    futures = [executor.submit(download_file, url, path) for url, path in tasks]
                    
                    # 等待所有任务完成
                    for future in futures:
                        future.result()

    print("\n=== 下载任务全部完成 ===")
    print(f"数据位于: {SAVE_ROOT}")
    print("提示：你可以使用 Hugging Face datasets 库直接加载这些本地 JSON/Parquet 文件。")

if __name__ == "__main__":
    main()