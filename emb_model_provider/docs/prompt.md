/spec.md 设计一个使用transformer库，部署all-miniLM-L12-v2嵌入模型，并使用FastAPI对外提供openai兼容的api接口，包括以下接口：
/v1/models
/v1/embeddings

该模块的目的是提供一个兼容openai接口的简单嵌入模型，通过localhost:9000/v1/embeddings接口调用。只需要给出基本的实现，并添加管理模块用于控制内存占用、批处理大小、上下文长度等

通过transformers加载模型时必须优先加载本地，且保存位置定义为D:\models\all-MiniLM-L12-v2，以免重复下载或下载到用户目录

在编写design、tasks文档的阶段自行使用context7 mcp和tavily查询相关信息