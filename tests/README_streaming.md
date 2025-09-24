# 流式TTS服务使用说明

## 概述

这是一个基于FastAPI的流式文本转语音（TTS）服务，支持将长文本切分后流式处理并返回音频数据。

## 功能特性

- ✅ **文本智能切分**: 支持按句子或固定长度切分长文本
- ✅ **流式音频生成**: 实时生成并返回音频片段
- ✅ **Web界面**: 提供美观的HTML演示页面
- ✅ **批量处理**: 支持一次性生成完整音频文件
- ✅ **自动播放**: 前端可自动播放生成的音频片段
- ✅ **进度显示**: 实时显示处理进度和状态
- ✅ **错误处理**: 完善的错误处理和状态反馈

## 文件说明

```
tests/
├── streaming_tts_server.py      # 流式TTS服务器
├── streaming_demo.html          # Web演示页面
├── test_streaming_client.py     # Python客户端测试脚本
├── start_streaming_server.py    # 服务启动脚本
├── README_streaming.md          # 本说明文档
└── sample_prompt.wav            # 音频样本文件
```

## 快速开始

### 1. 安装依赖

```bash
pip install fastapi uvicorn soundfile numpy pydantic aiohttp requests
```

### 2. 启动服务

#### 方法一：使用启动脚本（推荐）
```bash
python start_streaming_server.py
```

#### 方法二：直接启动
```bash
python streaming_tts_server.py --host 127.0.0.1 --port 8000
```

#### 方法三：模拟模式（用于测试）
```bash
python start_streaming_server.py --mock
```

### 3. 访问服务

- **Web演示页面**: 打开 `streaming_demo.html`
- **API文档**: http://127.0.0.1:8000/docs
- **健康检查**: http://127.0.0.1:8000/health

## API接口

### 1. 流式TTS接口

**POST** `/tts_streaming`

```json
{
    "text": "要转换的文本内容",
    "audio_paths": ["/path/to/sample.wav"],
    "seed": 8,
    "chunk_size": 100,
    "sentence_split": true
}
```

**响应**: Server-Sent Events (SSE) 流

```
data: {"chunk_index": 0, "total_chunks": 3, "text": "第一个句子", "timestamp": 1234567890}

data: {"type": "audio", "chunk_index": 0, "data": "base64编码的音频数据"}

data: {"type": "complete", "total_chunks": 3, "timestamp": 1234567890}
```

### 2. 批量TTS接口

**POST** `/tts_batch`

```json
{
    "text": "要转换的文本内容",
    "audio_paths": ["/path/to/sample.wav"],
    "seed": 8,
    "chunk_size": 100,
    "sentence_split": true
}
```

**响应**: 完整的WAV音频文件

### 3. 健康检查

**GET** `/health`

```json
{
    "status": "healthy",
    "timestamp": 1234567890,
    "tts_available": true
}
```

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `text` | string | - | 要转换的文本内容 |
| `audio_paths` | array | - | 音频样本文件路径列表 |
| `seed` | integer | 8 | 随机种子，影响音频生成的随机性 |
| `chunk_size` | integer | 100 | 文本切分的最大字符数 |
| `sentence_split` | boolean | true | 是否按句子切分（false为按长度切分） |

## 使用示例

### Python客户端示例

```python
import asyncio
import aiohttp
import json
import base64

async def test_streaming_tts():
    async with aiohttp.ClientSession() as session:
        data = {
            "text": "这是一个测试文本，用于演示流式TTS功能。",
            "audio_paths": ["/path/to/sample.wav"],
            "chunk_size": 50,
            "sentence_split": True
        }
        
        async with session.post(
            "http://127.0.0.1:8000/tts_streaming",
            json=data
        ) as response:
            async for line in response.content:
                line_str = line.decode('utf-8').strip()
                if line_str.startswith('data: '):
                    event_data = json.loads(line_str[6:])
                    
                    if event_data.get("type") == "audio":
                        # 处理音频数据
                        audio_bytes = base64.b64decode(event_data["data"])
                        with open(f"chunk_{event_data['chunk_index']}.wav", "wb") as f:
                            f.write(audio_bytes)

# 运行测试
asyncio.run(test_streaming_tts())
```

### JavaScript客户端示例

```javascript
async function streamingTTS() {
    const response = await fetch('/tts_streaming', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            text: "测试文本",
            audio_paths: ["/path/to/sample.wav"],
            chunk_size: 100,
            sentence_split: true
        })
    });
    
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    
    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        
        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');
        
        for (const line of lines) {
            if (line.startsWith('data: ')) {
                const data = JSON.parse(line.slice(6));
                
                if (data.type === 'audio') {
                    // 处理音频数据
                    const audioBlob = base64ToBlob(data.data, 'audio/wav');
                    const audioUrl = URL.createObjectURL(audioBlob);
                    
                    // 播放音频
                    const audio = new Audio(audioUrl);
                    audio.play();
                }
            }
        }
    }
}
```

## 测试方法

### 1. 命令行测试

```bash
# 运行Python客户端测试
python test_streaming_client.py
```

### 2. Web界面测试

1. 启动服务器
2. 打开 `streaming_demo.html`
3. 输入测试文本
4. 点击"开始流式生成"
5. 观察实时生成的音频片段

### 3. API测试

```bash
# 健康检查
curl http://127.0.0.1:8000/health

# 批量TTS测试
curl -X POST http://127.0.0.1:8000/tts_batch \
  -H "Content-Type: application/json" \
  -d '{
    "text": "测试文本",
    "audio_paths": ["/path/to/sample.wav"]
  }' \
  --output test_output.wav
```

## 配置说明

### 环境变量

- `MODEL_DIR`: TTS模型目录路径

### 命令行参数

```bash
python streaming_tts_server.py --help

usage: streaming_tts_server.py [-h] [--host HOST] [--port PORT] [--model_dir MODEL_DIR]

options:
  -h, --help            show this help message and exit
  --host HOST           Host address
  --port PORT           Port number  
  --model_dir MODEL_DIR TTS model directory
```

## 故障排除

### 常见问题

1. **服务启动失败**
   - 检查端口是否被占用
   - 确认依赖包已正确安装
   - 查看错误日志

2. **模型加载失败**
   - 确认模型目录路径正确
   - 检查 `config.yaml` 文件是否存在
   - 可使用 `--mock` 参数进行测试

3. **音频生成失败**
   - 检查音频样本文件路径是否正确
   - 确认音频样本格式为WAV
   - 查看服务器日志获取详细错误信息

4. **前端无法连接**
   - 确认服务器地址和端口配置正确
   - 检查CORS设置
   - 查看浏览器控制台错误信息

### 日志调试

服务器会输出详细的处理日志，包括：
- 文本切分结果
- 音频生成进度
- 错误信息和堆栈跟踪

## 性能优化

1. **文本切分优化**
   - 合理设置 `chunk_size` 参数
   - 根据内容特点选择切分模式

2. **内存使用优化**
   - 控制并发处理数量
   - 及时释放音频缓存

3. **网络传输优化**
   - 使用适当的音频压缩
   - 考虑实现音频流压缩

## 扩展开发

### 添加新的切分策略

```python
class CustomSplitter:
    @staticmethod
    def split_by_custom_rule(text: str, max_length: int) -> List[str]:
        # 实现自定义切分逻辑
        pass
```

### 集成其他TTS引擎

```python
class CustomTTSEngine:
    async def infer(self, audio_paths, text, **kwargs):
        # 实现自定义TTS推理逻辑
        pass
```

## 许可证

本项目遵循原项目的许可证要求。

## 贡献

欢迎提交Issue和Pull Request来改进这个项目。
