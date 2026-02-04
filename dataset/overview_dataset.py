#!/usr/bin/env python3
"""
RedPajama-Data-V2-Local 数据集概览脚本
打印各个子目录中第一个文件的基本信息
"""

import os
import gzip
import json
from pathlib import Path
from collections import defaultdict

def get_file_size_human(size_bytes):
    """将字节转换为人类可读的格式"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"

def count_lines_in_gz(filepath, max_lines=None):
    """统计压缩文件中的行数"""
    count = 0
    try:
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            for line in f:
                count += 1
                if max_lines and count >= max_lines:
                    return count, True  # 返回是否截断
    except Exception as e:
        return -1, False
    return count, False

def get_first_record(filepath):
    """获取文件的第一条记录"""
    try:
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            first_line = f.readline()
            if first_line:
                return json.loads(first_line)
    except Exception as e:
        return None
    return None

def analyze_directory(base_path, dir_type):
    """分析指定类型的目录"""
    print(f"\n{'='*80}")
    print(f"分析 {dir_type} 目录")
    print(f"{'='*80}\n")

    # 获取所有子目录
    subdirs = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])

    print(f"总共找到 {len(subdirs)} 个子目录\n")

    # 分析前10个和最后5个子目录
    sample_dirs = subdirs[:10] + (['...'] if len(subdirs) > 15 else []) + subdirs[-5:]

    total_size = 0
    total_records = 0

    for subdir in sample_dirs:
        if subdir == '...':
            print(f"\n{'.'*80}\n")
            continue

        subdir_path = os.path.join(base_path, subdir)

        # 获取第一个文件
        files = sorted([f for f in os.listdir(subdir_path) if f.endswith('.json.gz')])
        if not files:
            print(f"[{subdir}] 没有找到 .json.gz 文件")
            continue

        first_file = files[0]
        filepath = os.path.join(subdir_path, first_file)

        # 获取文件信息
        file_size = os.path.getsize(filepath)
        total_size += file_size

        # 统计行数（最多统计1000行以加快速度）
        line_count, truncated = count_lines_in_gz(filepath, max_lines=1000)
        if line_count > 0:
            total_records += line_count

        # 获取第一条记录
        first_record = get_first_record(filepath)

        print(f"[{subdir}] {first_file}")
        print(f"  文件大小: {get_file_size_human(file_size)}")

        if line_count >= 0:
            if truncated:
                print(f"  记录数: >{line_count} (已截断统计)")
            else:
                print(f"  记录数: {line_count}")
        else:
            print(f"  记录数: 读取失败")

        # 显示第一条记录的关键信息
        if first_record:
            if dir_type == "documents":
                print(f"  示例URL: {first_record.get('url', 'N/A')}")
                print(f"  文档长度: {first_record.get('length', 'N/A')} 字符")
                print(f"  文档行数: {first_record.get('nlines', 'N/A')}")
                title = first_record.get('title', 'N/A')
                if len(title) > 60:
                    title = title[:60] + "..."
                print(f"  标题: {title}")
            elif dir_type == "quality_signals":
                print(f"  示例ID: {first_record.get('id', 'N/A')}")
                metadata = first_record.get('metadata', {})
                print(f"  语言: {metadata.get('language', 'N/A')}")
                print(f"  来源域名: {metadata.get('source_domain', 'N/A')}")
                quality_signals = first_record.get('quality_signals', {})
                print(f"  质量信号数: {len(quality_signals)}")

        print()

    print(f"\n{'='*80}")
    print(f"统计摘要 ({dir_type})")
    print(f"{'='*80}")
    print(f"子目录总数: {len(subdirs)}")
    print(f"采样目录数: {len([d for d in sample_dirs if d != '...'])}")
    print(f"采样文件总大小: {get_file_size_human(total_size)}")
    if total_records > 0:
        print(f"采样记录总数: ~{total_records}")
    print()

def main():
    base_dir = "/mnt/shared-storage-user/puyuan/code/EBT/dataset/RedPajama-Data-V2-Local"

    print("="*80)
    print("RedPajama-Data-V2-Local 数据集概览")
    print("="*80)

    # 检查目录结构
    print("\n目录结构:")
    for item in sorted(os.listdir(base_dir)):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path) and not item.startswith('.'):
            print(f"  - {item}/")

    # 分析 listings
    listings_path = os.path.join(base_dir, "listings")
    if os.path.exists(listings_path):
        print(f"\n{'='*80}")
        print("分析 listings 目录")
        print(f"{'='*80}\n")
        for file in os.listdir(listings_path):
            filepath = os.path.join(listings_path, file)
            if os.path.isfile(filepath):
                size = os.path.getsize(filepath)
                print(f"  {file}")
                print(f"    大小: {get_file_size_human(size)}")
                # 显示前几行
                with open(filepath, 'r') as f:
                    lines = [f.readline().strip() for _ in range(3)]
                    print(f"    前3行:")
                    for line in lines:
                        if line:
                            print(f"      {line[:80]}{'...' if len(line) > 80 else ''}")
                print()

    # 分析 documents
    documents_path = os.path.join(base_dir, "documents/2023-06")
    if os.path.exists(documents_path):
        analyze_directory(documents_path, "documents")

    # 分析 quality_signals
    quality_signals_path = os.path.join(base_dir, "quality_signals/2023-06")
    if os.path.exists(quality_signals_path):
        analyze_directory(quality_signals_path, "quality_signals")

    print("\n" + "="*80)
    print("概览完成")
    print("="*80)

if __name__ == "__main__":
    main()
