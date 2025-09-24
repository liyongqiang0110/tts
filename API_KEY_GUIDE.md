# API Key 使用指南

## 概述

为了提高 TTS 服务的安全性，我们为 `api_server.py` 添加了 API key 验证功能。当启用 API key 验证后，所有的 API 请求都需要提供有效的 API key 才能访问。

## 启用 API Key 验证

有两种方式设置 API key：

### 方式1：命令行参数
```bash
python api_server.py --api_key "your_secret_api_key_here"
```

### 方式2：环境变量
```bash
# Linux/Mac
export TTS_API_KEY="your_secret_api_key_here"
python api_server.py

# Windows
set TTS_API_KEY="your_secret_api_key_here"
python api_server.py
```

## 客户端请求方式

客户端可以通过以下四种方式提供 API key：

### 1. Authorization 头（Bearer Token）
```bash
curl -X POST "http://localhost:11996/tts" \
  -H "Authorization: Bearer your_secret_api_key_here" \
  -H "Content-Type: application/json" \
  -d '{"text": "你好世界", "character": "speaker1"}'
```

### 2. X-API-Key 头
```bash
curl -X POST "http://localhost:11996/tts" \
  -H "X-API-Key: your_secret_api_key_here" \
  -H "Content-Type: application/json" \
  -d '{"text": "你好世界", "character": "speaker1"}'
```

### 3. API-Key 头
```bash
curl -X POST "http://localhost:11996/tts" \
  -H "API-Key: your_secret_api_key_here" \
  -H "Content-Type: application/json" \
  -d '{"text": "你好世界", "character": "speaker1"}'
```

### 4. 查询参数
```bash
curl -X POST "http://localhost:11996/tts?api_key=your_secret_api_key_here" \
  -H "Content-Type: application/json" \
  -d '{"text": "你好世界", "character": "speaker1"}'
```

## JavaScript 示例

### 使用 fetch API
```javascript
// 使用 Authorization 头
fetch('http://localhost:11996/tts', {
  method: 'POST',
  headers: {
    'Authorization': 'Bearer your_secret_api_key_here',
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    text: '你好世界',
    character: 'speaker1'
  })
});

// 使用 X-API-Key 头
fetch('http://localhost:11996/tts', {
  method: 'POST',
  headers: {
    'X-API-Key': 'your_secret_api_key_here',
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    text: '你好世界',
    character: 'speaker1'
  })
});

// 使用查询参数
fetch('http://localhost:11996/tts?api_key=your_secret_api_key_here', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    text: '你好世界',
    character: 'speaker1'
  })
});
```

## Python 客户端示例

```python
import requests
import json

# 方式1：使用 Authorization 头
headers = {
    'Authorization': 'Bearer your_secret_api_key_here',
    'Content-Type': 'application/json'
}

# 方式2：使用 X-API-Key 头
headers = {
    'X-API-Key': 'your_secret_api_key_here',
    'Content-Type': 'application/json'
}

# 方式3：使用查询参数
params = {'api_key': 'your_secret_api_key_here'}

data = {
    'text': '你好世界',
    'character': 'speaker1'
}

response = requests.post(
    'http://localhost:11996/tts',
    headers=headers,  # 或者使用 params=params
    json=data
)
```

## 受保护的接口

所有 API 接口都受到 API key 保护：
- `/health` - 健康检查
- `/tts` - 基础 TTS 接口
- `/tts_url` - URL 音频路径 TTS 接口
- `/audio/speech` - OpenAI 兼容接口
- `/audio/voices` - 获取可用声音列表
- `/tts_streaming` - 流式 TTS 接口

## 错误响应

当 API key 验证失败时，服务器会返回 HTTP 401 错误：

```json
{
  "detail": {
    "error": "Invalid or missing API key",
    "message": "Please provide a valid API key via Authorization header (Bearer token), X-API-Key header, API-Key header, or api_key query parameter"
  }
}
```

## 安全建议

1. **使用强密码**：API key 应该是随机生成的强密码，至少 32 个字符
2. **定期更换**：建议定期更换 API key
3. **环境变量**：在生产环境中，推荐使用环境变量而不是命令行参数
4. **HTTPS**：在生产环境中使用 HTTPS 来保护 API key 传输
5. **日志安全**：确保 API key 不会被记录到日志文件中

## 禁用 API Key 验证

如果不设置 API key（既不通过命令行参数也不通过环境变量），则 API key 验证功能会被禁用，所有请求都会被允许访问。

```bash
# 不设置 API key，验证功能禁用
python api_server.py
```

启动时会显示警告信息：
```
警告: 未设置API密钥，所有请求都将被允许访问
```