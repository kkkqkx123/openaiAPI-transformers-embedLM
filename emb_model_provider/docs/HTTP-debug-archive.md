# HTTP Debug Archive

本文档记录了使用hurl工具对Embedding Model Provider API进行的HTTP调试测试。

## 测试环境

- **API服务器**: `localhost:9000`
- **测试工具**: hurl 7.0.0
- **测试时间**: 2025-12-12 15:37 (UTC)
- **项目版本**: 0.1.0

## HURL测试文件

测试文件位于 `tests/hurl/` 目录下：

1. `health.hurl` - 健康检查端点
2. `root.hurl` - 根端点
3. `models.hurl` - 模型列表端点
4. `embeddings.hurl` - 嵌入生成端点
5. `performance.hurl` - 性能监控端点

### 1. 健康检查端点

**HURL命令**:
```bash
hurl tests/hurl/health.hurl
```

**预期结果**:
- HTTP状态码 200
- 响应体包含 `{"status": "healthy"}`

**实际结果**:
- ✅ 测试通过
- 响应状态码 200
- 响应体: `{"status":"healthy"}`

### 2. 根端点

**HURL命令**:
```bash
hurl tests/hurl/root.hurl
```

**预期结果**:
- HTTP状态码 200
- 响应体包含API基本信息，包括版本和端点列表

**实际结果**:
- ✅ 测试通过
- 响应状态码 200
- 响应体: `{"message":"Embedding Model Provider API","version":"0.1.0","endpoints":["/v1/embeddings","/v1/models","/health"]}`

### 3. 模型列表端点

**HURL命令**:
```bash
hurl tests/hurl/models.hurl
```

**预期结果**:
- HTTP状态码 200
- 响应体包含模型列表，至少有一个模型
- 每个模型具有 `id` 和 `object` 字段

**实际结果**:
- ✅ 测试通过
- 响应状态码 200
- 响应体包含8个模型，包括默认模型和别名（如"default", "mini", "qwen3"等）
- 所有模型均包含 `id`, `object`, `created`, `owned_by` 字段

### 4. 嵌入生成端点

**HURL命令**:
```bash
hurl tests/hurl/embeddings.hurl
```

**预期结果**:
- HTTP状态码 200
- 响应体包含嵌入向量数据
- 嵌入向量维度为384（根据模型配置）
- 包含使用量统计

**实际结果**:
- ❌ 测试失败
- 响应状态码 500
- 错误信息: `{"error":{"message":"Internal server error","type":"internal_server_error","param":null,"code":"500"}}`
- 服务器日志显示: `EmbeddingService.generate_embeddings() missing 1 required positional argument: 'model_alias'`
- 问题: 动态批处理处理器调用 `generate_embeddings` 时缺少 `model_alias` 参数，导致内部服务器错误。

**额外测试**:
使用模型别名 "default" 进行测试:
```bash
curl -X POST http://localhost:9000/v1/embeddings -H "Content-Type: application/json" -d '{"input": "Hello world", "model": "default", "encoding_format": "float"}'
```
结果: 同样返回500错误，相同原因。

### 5. 性能监控端点

**HURL命令**:
```bash
hurl tests/hurl/performance.hurl
```

**预期结果**:
- 第一个GET请求返回性能指标
- 第二个POST请求重置性能指标并返回成功消息

**实际结果**:
- ⚠️ 部分通过
- GET请求: 响应状态码200，但JSON字段 `total_requests` 不存在（实际字段为 `total_requests_processed`）。更新hurl文件后测试通过。
- POST请求: ✅ 测试通过，返回 `{"message": "Performance metrics reset successfully"}`

更新后的hurl文件使用正确的JSON路径后，性能端点测试完全通过。

## 执行步骤

1. 确保API服务器正在运行。如果没有，使用以下命令启动：
   ```bash
   uv run python -m emb_model_provider.main
   ```

2. 在另一个终端中执行hurl测试。

## 问题与观察

### 发现的问题

1. **嵌入端点内部错误**:
   - 根本原因: `EmbeddingService.generate_embeddings()` 方法需要 `model_alias` 参数，但动态批处理处理器调用时未提供。
   - 影响: 所有嵌入请求（无论使用哪个模型）均返回500内部服务器错误。
   - 建议: 修复 `realtime_batch_processor.py` 中的调用，传递正确的 `model_alias` 参数。

2. **性能端点JSON字段不匹配**:
   - 预期字段 `total_requests` 不存在，实际字段为 `total_requests_processed`。
   - 已通过更新hurl测试文件修复。

3. **模型别名支持**:
   - 模型列表端点正确返回了配置的别名（如"default", "mini", "qwen3"等），表明模型映射配置生效。

### 其他观察

- 健康检查、根端点和模型列表端点工作正常。
- 服务器启动时成功加载了默认模型（sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2）。
- 动态批处理功能已启用（根据配置 `enable_dynamic_batching: true`），但存在参数传递错误。

## 结论

- 4个端点中有3个（健康检查、根端点、模型列表）工作正常。
- 嵌入端点因内部代码错误而失败，需要开发团队修复。
- 性能端点工作正常，但需要更新测试断言以匹配实际响应字段。

**建议行动**:
1. 修复 `EmbeddingService.generate_embeddings()` 调用缺失参数的问题。
2. 验证修复后重新运行hurl测试。
3. 考虑添加更多hurl测试用例，涵盖错误输入、批量输入、不同编码格式等场景。

本次HTTP调试测试成功识别了关键问题，为后续修复提供了明确方向。