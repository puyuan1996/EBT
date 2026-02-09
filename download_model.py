# # 在有网的电脑上运行
from huggingface_hub import snapshot_download

# 下载模型到当前目录的 models 文件夹下
snapshot_download(
    repo_id="EleutherAI/gpt-neox-20b",
    local_dir="./gpt-neox-20b",
    local_dir_use_symlinks=False  # 确保下载的是实际文件而不是软链接
)

# from huggingface_hub import snapshot_download

# # 指定下载目录
# local_dir = "./gpt-neox-20b-tokenizer"

# snapshot_download(
#     repo_id="EleutherAI/gpt-neox-20b",
#     local_dir=local_dir,
#     # 关键点：只下载 json 和 txt 文件，排除巨大的权重文件
#     allow_patterns=["*.json", "*.txt", "*.model"],
#     local_dir_use_symlinks=False  # 确保下载的是实体文件
# )

# print(f"下载完成！请将 {local_dir} 文件夹上传到你的服务器。")