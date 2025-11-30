from transformers import AutoTokenizer, AutoModel

# 你的本地模型路径
model_path = "D:\\models\\jina-embeddings-v2-base-code"

# 关键：必须保留 trust_remote_code=True（和下载时一致）
try:
    print("加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True  # 强制仅使用本地文件，不联网（避免远程补充）
    )
    
    print("加载模型...")
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True  # 强制仅用本地文件，验证本地权重和代码是否匹配
    )
    
    print("✅ 模型和 Tokenizer 加载成功，无权重不匹配警告！")
    print(f"📌 模型架构：{model.__class__.__name__}（应为 JinaBertModel 或 JinaEmbeddingsModel）")
except Exception as e:
    print(f"❌ 加载失败，权重与代码不匹配：{e}")