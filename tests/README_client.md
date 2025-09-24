# 流式TTS客户端使用说明

## 概述

这是一个针对已部署TTS服务的流式客户端，能够将长文本智能切分后流式处理，实现边生成边播放的效果。

## 功能特性

- ✅ **智能文本切分**: 支持按句子或固定长度切分长文本
- ✅ **流式处理**: 依次处理文本片段，避免长时间等待
- ✅ **实时播放**: 生成完成的音频片段可立即播放
- ✅ **进度显示**: 实时显示处理进度和状态
- ✅ **Web界面**: 美观的HTML界面，支持参数配置
- ✅ **错误处理**: 完善的错误处理和重试机制
- ✅ **批量播放**: 支持按顺序播放所有生成的音频片段

## 文件说明

```
tests/
├── streaming_client.py           # Python流式客户端核心库
├── streaming_web_client.html     # Web界面客户端
├── simple_streaming_test.py      # 简单测试脚本
└── README_client.md              # 本说明文档
```

## 快速开始

### 1. 确保服务端运行

确保你的TTS服务端已启动并可访问：
```bash
# 测试服务端是否可用
curl -X POST http://127.0.0.1:11996/tts_url \
  -H "Content-Type: application/json" \
  -d '{"text": "测试", "audio_paths": ["/path/to/sample.wav"]}'
```

### 2. 安装依赖

```bash
pip install aiohttp asyncio
```

### 3. 使用方式

#### 方式一：Web界面（推荐）

1. 直接打开 `streaming_web_client.html`
2. 配置服务器地址和参数
3. 输入要转换的文本
4. 点击"开始流式生成"
5. 实时查看处理进度和播放音频

#### 方式二：Python脚本

```bash
# 运行简单测试
python simple_streaming_test.py

# 或使用完整的客户端库
python -c "
import asyncio
from streaming_client import StreamingTTSClient

async def test():
    async with StreamingTTSClient('http://127.0.0.1:11996') as client:
        results = await client.process_text_streaming(
            text='你的长文本内容',
            audio_paths=['/path/to/sample.wav'],
            chunk_size=80,
            sentence_split=True
        )
        print(f'生成了 {len(results)} 个音频片段')

asyncio.run(test())
"
```

## 配置参数

### 服务器设置
- **服务器地址**: TTS服务的URL地址（默认：`http://127.0.0.1:11996`）
- **音频样本路径**: 用于语音克隆的样本文件路径

### 文本处理设置
- **切分大小**: 每个文本片段的最大字符数（建议：60-120）
- **按句子切分**: 是否优先按句子边界切分（推荐开启）

### 处理设置
- **并发处理**: 是否允许并发处理多个片段（当前为顺序处理）
- **重试次数**: 处理失败时的重试次数

## API使用示例

### Python客户端示例

```python
import asyncio
from streaming_client import StreamingTTSClient, AudioPlayer

async def streaming_tts_example():
    text = "这是一个很长的文本内容，需要进行流式处理..."
    audio_paths = ["/path/to/your/sample.wav"]
    
    # 创建客户端
    async with StreamingTTSClient("http://127.0.0.1:11996") as client:
        # 定义回调函数
        async def on_chunk_complete(index, result):
            print(f"完成片段 {index}: {result['text'][:30]}...")
            # 这里可以立即播放或保存音频
            filename = f"output_{index:03d}.wav"
            with open(filename, "wb") as f:
                f.write(result['audio_data'])
        
        # 开始流式处理
        results = await client.process_text_streaming(
            text=text,
            audio_paths=audio_paths,
            chunk_size=80,
            sentence_split=True,
            on_chunk_complete=on_chunk_complete
        )
        
        print(f"处理完成，生成了 {len(results)} 个音频片段")

# 运行示例
asyncio.run(streaming_tts_example())
```

### 文本切分示例

```python
from streaming_client import TextSplitter

# 按句子切分
text = "第一句话。第二句话！第三句话？"
chunks = TextSplitter.split_by_sentence(text, max_length=50)
print(chunks)  # ['第一句话。', '第二句话！', '第三句话？']

# 按长度切分
chunks = TextSplitter.split_by_length(text, max_length=10)
print(chunks)  # ['第一句话。第二句话', '！第三句话？']
```

## Web界面使用

### 界面功能

1. **文本输入区域**
   - 支持多行文本输入
   - 显示字符数统计

2. **参数配置**
   - 服务器地址设置
   - 切分参数调整
   - 音频样本路径配置

3. **处理控制**
   - 开始/停止处理按钮
   - 清空结果功能
   - 实时进度显示

4. **结果展示**
   - 每个片段的处理状态
   - 音频大小和处理时间
   - 单独播放按钮

5. **全局播放控制**
   - 顺序播放所有片段
   - 播放状态显示
   - 播放进度提示

### 状态指示器

- 🟡 **处理中**: 片段正在生成音频
- 🟢 **已完成**: 片段处理成功，可以播放
- 🔴 **处理失败**: 片段处理出现错误

## 性能优化建议

### 文本切分优化

1. **合理设置切分大小**
   - 短文本（<50字符）：不需要切分
   - 中等文本（50-200字符）：chunk_size=80-120
   - 长文本（>200字符）：chunk_size=60-100

2. **选择切分模式**
   - 句子切分：适合正式文本，保持语义完整性
   - 长度切分：适合连续文本，处理速度更快

### 网络优化

1. **服务器部署**
   - 确保服务器有足够的处理能力
   - 考虑使用负载均衡

2. **客户端优化**
   - 适当的片段间延迟（避免服务器过载）
   - 实现重试机制（处理网络异常）

### 音频处理优化

1. **音频格式**
   - 使用适当的音频采样率和比特率
   - 考虑音频压缩以减少传输时间

2. **缓存策略**
   - 缓存已生成的音频片段
   - 避免重复处理相同文本

## 故障排除

### 常见问题

1. **连接服务器失败**
   ```
   ❌ 无法连接到服务器
   ```
   - 检查服务器地址是否正确
   - 确认服务器是否启动
   - 检查网络连接

2. **音频样本路径错误**
   ```
   ❌ 处理失败: 音频文件不存在
   ```
   - 确认音频样本文件存在
   - 检查文件路径是否正确
   - 确保服务器有访问权限

3. **文本切分异常**
   ```
   ❌ 文本切分失败
   ```
   - 检查文本内容是否包含特殊字符
   - 调整切分参数
   - 尝试使用长度切分模式

4. **播放失败**
   ```
   ❌ 音频播放失败
   ```
   - 检查浏览器音频权限
   - 确认音频格式支持
   - 尝试下载文件手动播放

### 调试方法

1. **启用详细日志**
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **检查服务器响应**
   ```bash
   curl -v -X POST http://127.0.0.1:11996/tts_url \
     -H "Content-Type: application/json" \
     -d '{"text": "测试", "audio_paths": ["/path/to/sample.wav"]}'
   ```

3. **测试单个片段**
   - 使用短文本测试基本功能
   - 逐步增加文本长度
   - 检查每个处理阶段

## 扩展开发

### 添加新的切分策略

```python
class CustomTextSplitter:
    @staticmethod
    def split_by_keywords(text: str, keywords: list, max_length: int) -> list:
        """基于关键词的智能切分"""
        # 实现自定义切分逻辑
        pass
```

### 集成音频播放库

```python
import pygame
import sounddevice as sd

class AdvancedAudioPlayer:
    def __init__(self):
        pygame.mixer.init()
    
    def play_with_effects(self, audio_data, effects=None):
        # 添加音频特效处理
        pass
```

### 实现音频缓存

```python
import hashlib
import pickle

class AudioCache:
    def __init__(self, cache_dir="audio_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def get_cache_key(self, text, audio_paths):
        content = f"{text}:{':'.join(audio_paths)}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, key):
        cache_file = self.cache_dir / f"{key}.cache"
        if cache_file.exists():
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        return None
    
    def set(self, key, data):
        cache_file = self.cache_dir / f"{key}.cache"
        with open(cache_file, "wb") as f:
            pickle.dump(data, f)
```

## 许可证

本项目遵循原项目的许可证要求。

## 贡献

欢迎提交Issue和Pull Request来改进这个客户端。
