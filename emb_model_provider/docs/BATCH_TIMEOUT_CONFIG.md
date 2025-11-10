# 批处理超时机制及配置说明

## 用户使用说明

### 配置批处理超时参数

在 `.env` 文件中，您可以配置以下参数来控制批处理的超时行为：

```bash
# 动态批处理最大等待时间（毫秒）
# 系统会等待指定时间以收集更多请求形成更大的批次
EMB_PROVIDER_MAX_WAIT_TIME_MS=100

# 硬超时附加时间（秒）
# 在max_wait_time之后额外等待的时间，用于强制处理小批次，避免长时间挂起
EMB_PROVIDER_HARD_TIMEOUT_ADDITIONAL_SECONDS=1.0

# 动态批处理最小批次大小
# 即使等待时间未到，达到最小批次大小也会开始处理
EMB_PROVIDER_MIN_BATCH_SIZE=12

# 最大批处理大小（同时处理的最大文本数量）
EMB_PROVIDER_MAX_BATCH_SIZE=32

# 启用动态批处理（根据请求到达情况动态调整批处理大小）
EMB_PROVIDER_ENABLE_DYNAMIC_BATCHING=true
```

### 配置说明

1. **正常处理流程**：
   - 当请求数量 ≥ `EMB_PROVIDER_MAX_BATCH_SIZE` 时，立即处理
   - 当请求数量 ≥ `EMB_PROVIDER_MIN_BATCH_SIZE` 且等待时间 ≥ `EMB_PROVIDER_MAX_WAIT_TIME_MS` 时，立即处理

2. **超时处理机制**：
   - 当请求数量 < `EMB_PROVIDER_MIN_BATCH_SIZE` 且等待时间 ≥ `EMB_PROVIDER_MAX_WAIT_TIME_MS` 时，继续等待
   - 当请求数量 < `EMB_PROVIDER_MIN_BATCH_SIZE` 且等待时间 ≥ (`EMB_PROVIDER_MAX_WAIT_TIME_MS` + `EMB_PROVIDER_HARD_TIMEOUT_ADDITIONAL_SECONDS`) 时，强制处理所有现有请求

3. **推荐配置**：
   - **高吞吐量场景**：增加 `EMB_PROVIDER_MAX_WAIT_TIME_MS` 和 `EMB_PROVIDER_MIN_BATCH_SIZE`，提高批处理效率
   - **低延迟场景**：减少 `EMB_PROVIDER_MAX_WAIT_TIME_MS` 和 `EMB_PROVIDER_HARD_TIMEOUT_ADDITIONAL_SECONDS`，降低响应延迟
   - **低流量场景**：适当减少 `EMB_PROVIDER_MIN_BATCH_SIZE` 和 `EMB_PROVIDER_HARD_TIMEOUT_ADDITIONAL_SECONDS`，避免长时间等待

---

## 超时机制详解

### 1. 机制概述

批处理超时机制是嵌入模型服务中的一个重要组件，旨在平衡处理效率和响应延迟。该机制通过以下策略实现：

- **动态批处理**：收集多个请求一起处理，提高GPU利用率和吞吐量
- **自适应超时**：根据系统负载情况动态调整等待时间
- **双重保护**：防止在低流量情况下出现长时间挂起

### 2. 配置参数详解

#### 2.1 `EMB_PROVIDER_MAX_WAIT_TIME_MS` (默认: 100ms)
- **功能**: 控制在未达到最小批处理大小时的等待时间
- **作用**: 当请求队列中的请求数量少于最小批处理大小时，系统会等待最多 `max_wait_time_ms` 毫秒，以期望收集更多请求
- **影响**: 
  - 值越大，等待时间越长，可能获得更大的批次，提高吞吐量，但响应延迟增加
  - 值越小，等待时间越短，响应延迟降低，但批处理效率可能下降

#### 2.2 `EMB_PROVIDER_HARD_TIMEOUT_ADDITIONAL_SECONDS` (默认: 1.0秒)
- **功能**: 硬超时机制，作为后备保障
- **作用**: 在 `max_wait_time_ms` 基础上额外等待的时间，保证即使请求数量未达到最小批处理要求，也不会无限等待
- **影响**:
  - 提供响应时间的上界保障，避免长时间挂起
  - 防止在低流量场景下请求长时间得不到处理

#### 2.3 `EMB_PROVIDER_MIN_BATCH_SIZE` (默认: 1)
- **功能**: 最小批处理大小阈值
- **作用**: 当请求数量达到此值且等待时间超过 `max_wait_time_ms` 时，立即处理
- **影响**: 
  - 值越大，要求更多的请求才能触发处理，提高批处理效率，但可能增加延迟
  - 值越小，响应更快，但批处理效率可能降低

#### 2.4 `EMB_PROVIDER_MAX_BATCH_SIZE` (默认: 32)
- **功能**: 最大批处理大小限制
- **作用**: 当队列中请求数量达到此值时，立即处理，不考虑等待时间
- **影响**: 直接限制批处理大小，避免单次处理过大批次导致的内存或性能问题

### 3. 完整处理逻辑

1. **请求到达**:
   - 新请求被添加到批处理队列
   - 记录请求到达时间戳

2. **立即处理条件**:
   - 队列请求数量 ≥ `max_batch_size`：立即处理并清空队列

3. **定时检查逻辑**:
   - 检查队列中最老请求的等待时间
   - 如果等待时间 ≥ `max_wait_time_ms` 且请求数量 ≥ `min_batch_size`：处理批次
   - 如果等待时间 ≥ (`max_wait_time_ms` + `hard_timeout_additional_seconds`)：强制处理所有现有请求

4. **事件驱动机制**:
   - 实现采用事件驱动架构，减少不必要的轮询
   - 当有新请求到达时，立即检查是否满足处理条件
   - 智能计算下次检查时间，仅在必要时唤醒处理循环

### 4. 性能优化

1. **事件驱动架构**:
   - 使用 `asyncio.Event` 替代固定间隔轮询
   - 只在有新请求或达到超时时间时才检查处理条件

2. **智能超时计算**:
   - 动态计算下次检查时间，基于最老请求的超时时间
   - 减少不必要的 CPU 周期消耗

3. **线程安全**:
   - 使用锁保护共享资源访问
   - 确保多线程环境下的数据一致性

### 5. 配置建议

#### 5.1 高吞吐量场景 (如批处理任务)
```bash
EMB_PROVIDER_MAX_WAIT_TIME_MS=200
EMB_PROVIDER_HARD_TIMEOUT_ADDITIONAL_SECONDS=2.0
EMB_PROVIDER_MIN_BATCH_SIZE=16
EMB_PROVIDER_MAX_BATCH_SIZE=64
EMB_PROVIDER_ENABLE_DYNAMIC_BATCHING=true
```

#### 5.2 低延迟场景 (如实时API服务)
```bash
EMB_PROVIDER_MAX_WAIT_TIME_MS=50
EMB_PROVIDER_HARD_TIMEOUT_ADDITIONAL_SECONDS=0.5
EMB_PROVIDER_MIN_BATCH_SIZE=2
EMB_PROVIDER_MAX_BATCH_SIZE=16
EMB_PROVIDER_ENABLE_DYNAMIC_BATCHING=true
```

#### 5.3 低流量场景 (如测试或小规模应用)
```bash
EMB_PROVIDER_MAX_WAIT_TIME_MS=50
EMB_PROVIDER_HARD_TIMEOUT_ADDITIONAL_SECONDS=0.5
EMB_PROVIDER_MIN_BATCH_SIZE=1
EMB_PROVIDER_MAX_BATCH_SIZE=8
EMB_PROVIDER_ENABLE_DYNAMIC_BATCHING=true
```

### 6. 注意事项

1. **配置平衡**:
   - `max_wait_time_ms` 和 `min_batch_size` 需要平衡设置
   - 过高的值可能导致响应延迟过大
   - 过低的值可能无法充分利用批处理优势

2. **硬件适配**:
   - 系统会根据GPU内存自动调整批处理参数
   - 高端GPU（≥16GB）会自动增加等待时间以获得更大的批次
   - 低端GPU会减少等待时间以快速处理请求

3. **监控与调优**:
   - 通过性能监控端点（/v1/performance）观察批处理效果
   - 根据实际业务场景调整参数
   - 关注平均延迟、吞吐量和批处理效率等指标