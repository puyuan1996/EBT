#!/usr/bin/env python3
"""
查看 RedPajama 数据集文件的详细内容
"""

import gzip
import json
import sys

def inspect_file(filepath, num_lines=5):
    """检查文件并打印详细信息"""

    print(f"文件路径: {filepath}")
    print("="*80)

    # 统计总行数
    print("\n正在统计总行数...")
    total_lines = 0
    try:
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            for line in f:
                total_lines += 1
        print(f"✓ 总行数: {total_lines:,}")
    except Exception as e:
        print(f"✗ 统计失败: {e}")
        return

    # 读取并显示前几行
    print(f"\n前 {num_lines} 行内容:")
    print("="*80)

    try:
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            for i in range(num_lines):
                line = f.readline()
                if not line:
                    break

                print(f"\n--- 第 {i+1} 行 ---")
                try:
                    data = json.loads(line)

                    # 打印关键字段
                    print(f"URL: {data.get('url', 'N/A')}")
                    print(f"标题: {data.get('title', 'N/A')}")
                    print(f"下载日期: {data.get('date_download', 'N/A')}")
                    print(f"来源域名: {data.get('source_domain', 'N/A')}")
                    print(f"语言: {data.get('language', 'N/A')}")
                    print(f"语言分数: {data.get('language_score', 'N/A')}")
                    print(f"文档长度: {data.get('length', 'N/A')} 字符")
                    print(f"文档行数: {data.get('nlines', 'N/A')}")
                    print(f"原始长度: {data.get('original_length', 'N/A')} 字符")
                    print(f"原始行数: {data.get('original_nlines', 'N/A')}")
                    print(f"困惑度: {data.get('perplexity', 'N/A')}")
                    print(f"分桶: {data.get('bucket', 'N/A')}")

                    # 显示内容预览
                    raw_content = data.get('raw_content', '')
                    if raw_content:
                        preview_length = 300
                        preview = raw_content[:preview_length]
                        if len(raw_content) > preview_length:
                            preview += "..."
                        print(f"\n内容预览 (前{preview_length}字符):")
                        print("-" * 80)
                        print(preview)
                        print("-" * 80)

                except json.JSONDecodeError as e:
                    print(f"JSON 解析错误: {e}")
                    print(f"原始内容: {line[:200]}...")

    except Exception as e:
        print(f"读取失败: {e}")

    print("\n" + "="*80)
    print("检查完成")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python3 inspect_file.py <文件路径> [行数]")
        print("\n示例:")
        print("  python3 inspect_file.py /path/to/file.json.gz")
        print("  python3 inspect_file.py /path/to/file.json.gz 10")
        sys.exit(1)

    filepath = sys.argv[1]
    num_lines = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    inspect_file(filepath, num_lines)
