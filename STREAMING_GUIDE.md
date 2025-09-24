# 实时流式TTS音频生成使用指南

## 🎵 功能概述

本系统实现了完整的实时流式TTS音频生成和播放方案，支持：

- ✅ **智能文本分割**: 按自定义标点符号分割长文本
- ✅ **实时音频生成**: 分块生成音频片段，无需等待全量完成
- ✅ **流式播放**: 前端实时接收并播放音频流
- ✅ **音频队列管理**: 支持顺序播放、暂停控制
- ✅ **可视化界面**: 实时显示生成进度和状态

## 🚀 快速开始

### 1. 启动后端服务

```bash
cd index-tts-vllm
python api_server.py --host 0.0.0.0 --port 11996 --model_dir /path/to/your/model
```

### 2. 打开前端页面

在浏览器中打开 `realtime_audio_streaming.html` 文件

### 3. 配置参数

- **服务器地址**: 后端API服务地址 (默认: http://localhost:11996)
- **音频样本路径**: 参考音频文件路径
- **分块大小**: 文本分割的最大字符数 (建议: 50-200)
- **随机种子**: 影响音频生成的随机性
- **自定义分割符**: 可选，留空使用默认标点符号

### 4. 开始生成

1. 在文本框中输入要转换的内容
2. 点击"🚀 开始流式生成"
3. 观察实时生成进度
4. 音频片段生成后自动播放（或手动控制）

## 📡 API接口说明

### POST /tts_streaming

流式TTS音频生成接口

**请求参数:**
```json
{
    "text": "要转换的文本内容",
    "audio_paths": ["/path/to/sample.wav"],
    "seed": 8,
    "chunk_size": 100,
    "custom_punctuation": "，。！？"
}
```

**响应格式:** Server-Sent Events (SSE)

```
data: {"type": "start", "total_chunks": 3, "chunks": ["文本1", "文本2", "文本3"]}

data: {"type": "chunk_start", "chunk_index": 0, "text": "文本1", "total_chunks": 3}

data: {"type": "audio", "chunk_index": 0, "data": "base64音频数据", "processing_time": 1.2, "size": 12345}

data: {"type": "complete", "total_chunks": 3}
```

## 🔧 核心特性

### 智能文本分割

```python
# 默认按中文标点分割
chunks = TextSplitter.split_by_punctuation(text, max_length=100)

# 自定义分割符
chunks = TextSplitter.split_by_punctuation(text, max_length=100, custom_punctuation="，。！？")
```

支持的默认标点符号：
- 中文：`。！？；…`
- 换行符：`\n`

### 流式音频处理

1. **文本预处理**: 智能分割成合适大小的片段
2. **并发生成**: 逐个片段调用TTS引擎
3. **实时推送**: 通过SSE流式推送音频数据
4. **前端缓存**: 音频片段在前端缓存并排队播放

### 音频播放控制

- **自动播放**: 片段生成完成后自动按顺序播放
- **手动控制**: 可以播放任意指定片段
- **音量控制**: 全局音量调节
- **播放状态**: 实时显示当前播放状态

## 📊 性能优化

### 后端优化

1. **异步处理**: 使用asyncio实现非阻塞音频生成
2. **内存管理**: 及时释放音频缓存
3. **错误处理**: 完善的异常处理和恢复机制
4. **限流控制**: 避免服务器压力过大

### 前端优化

1. **音频预加载**: 提前加载下一个音频片段
2. **内存释放**: 播放完成后释放Blob URL
3. **UI响应**: 异步处理，避免界面卡顿
4. **错误重试**: 自动重试失败的音频片段

## 🛠 配置说明

### 服务器配置

```bash
python api_server.py \
    --host 0.0.0.0 \
    --port 11996 \
    --model_dir /path/to/IndexTeam/Index-TTS \
    --gpu_memory_utilization 0.25
```

### 客户端配置

在 `realtime_audio_streaming.html` 中修改默认配置：

```javascript
// 服务器地址
this.serverUrlInput = "http://localhost:11996";

// 默认分块大小
this.chunkSizeInput = 100;

// 默认音量
this.globalVolume = 0.8;
```

## 🔍 故障排除

### 常见问题

1. **服务连接失败**
   - 检查服务器地址和端口
   - 确认服务是否正常启动
   - 查看浏览器控制台错误信息

2. **音频生成失败**
   - 检查音频样本路径是否正确
   - 确认模型文件是否存在
   - 查看服务器日志错误信息

3. **音频播放问题**
   - 检查浏览器音频权限
   - 确认音频格式支持
   - 尝试手动播放测试

4. **性能问题**
   - 调整分块大小参数
   - 减少并发处理数量
   - 检查系统资源使用情况

### 调试技巧

1. **开启详细日志**
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **浏览器调试**
   - 打开开发者工具
   - 查看Network面板的SSE连接
   - 检查Console面板的错误信息

3. **性能监控**
   - 观察服务器CPU/内存使用
   - 监控音频生成时间
   - 检查网络传输延迟

## 📈 扩展开发

### 添加新的分割策略

```python
class CustomSplitter:
    @staticmethod
    def split_by_semantic(text: str, max_length: int) -> List[str]:
        # 实现语义分割逻辑
        pass
```

### 集成其他TTS引擎

```python
class CustomTTSEngine:
    async def infer(self, audio_paths, text, **kwargs):
        # 实现自定义TTS推理
        pass
```

### 添加音频后处理

```python
def post_process_audio(audio_bytes: bytes) -> bytes:
    # 音频降噪、音量标准化等
    return processed_audio_bytes
```

## 📝 更新日志

### v1.0.0 (2024-12-19)
- ✅ 实现基础流式TTS功能
- ✅ 支持自定义标点分割
- ✅ 完整的前端播放界面
- ✅ 音频队列管理
- ✅ 实时状态显示

## 📄 许可证

本项目遵循原项目的许可证要求。
