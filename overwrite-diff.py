from transformers import AutoModel, AutoTokenizer

model_name = "jinaai/jina-embeddings-v2-base-code"
model_path = "D:\\models\\jina-embeddings-v2-base-code"

# 直接用官方名称加载，自动覆盖不匹配的本地文件（仅下载差异部分）
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    cache_dir=model_path,
    local_files_only=False
)
model = AutoModel.from_pretrained(
    model_name,
    trust_remote_code=True,
    cache_dir=model_path,
    local_files_only=False
)
