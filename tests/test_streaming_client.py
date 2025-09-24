#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
流式TTS客户端测试脚本
"""

import asyncio
import aiohttp
import json
import base64
import time
from pathlib import Path

class StreamingTTSClient:
    """流式TTS客户端"""
    
    def __init__(self, server_url="http://127.0.0.1:8000"):
        self.server_url = server_url.rstrip('/')
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def test_health(self):
        """测试健康检查"""
        try:
            async with self.session.get(f"{self.server_url}/health") as response:
                data = await response.json()
                print(f"健康检查: {data}")
                return response.status == 200
        except Exception as e:
            print(f"健康检查失败: {e}")
            return False
    
    async def stream_tts(self, text, audio_paths, **kwargs):
        """流式TTS"""
        request_data = {
            "text": text,
            "audio_paths": audio_paths,
            "seed": kwargs.get("seed", 8),
            "chunk_size": kwargs.get("chunk_size", 100),
            "sentence_split": kwargs.get("sentence_split", True)
        }
        
        print(f"发送流式TTS请求...")
        print(f"文本长度: {len(text)} 字符")
        print(f"音频路径: {audio_paths}")
        print(f"设置: {json.dumps(request_data, ensure_ascii=False, indent=2)}")
        print("-" * 50)
        
        try:
            async with self.session.post(
                f"{self.server_url}/tts_streaming",
                json=request_data,
                headers={"Content-Type": "application/json"}
            ) as response:
                
                if response.status != 200:
                    error_text = await response.text()
                    print(f"请求失败: {response.status} - {error_text}")
                    return
                
                audio_chunks = []
                chunk_count = 0
                
                async for line in response.content:
                    line_str = line.decode('utf-8').strip()
                    
                    if line_str.startswith('data: '):
                        try:
                            data = json.loads(line_str[6:])
                            
                            if data.get("type") == "audio":
                                # 处理音频数据
                                chunk_index = data["chunk_index"]
                                audio_b64 = data["data"]
                                
                                # 解码音频数据
                                audio_bytes = base64.b64decode(audio_b64)
                                
                                # 保存音频片段
                                output_file = f"output_chunk_{chunk_index:03d}.wav"
                                with open(output_file, "wb") as f:
                                    f.write(audio_bytes)
                                
                                audio_chunks.append({
                                    "index": chunk_index,
                                    "file": output_file,
                                    "size": len(audio_bytes)
                                })
                                
                                print(f"✓ 接收音频片段 {chunk_index}: {len(audio_bytes)} 字节 -> {output_file}")
                                chunk_count += 1
                                
                            elif data.get("type") == "complete":
                                total_chunks = data["total_chunks"]
                                print(f"\n🎉 流式处理完成！总共生成 {total_chunks} 个音频片段")
                                break
                                
                            elif data.get("type") == "error":
                                print(f"❌ 处理错误: {data}")
                                
                            elif "chunk_index" in data and "text" in data:
                                # 处理元数据
                                chunk_index = data["chunk_index"]
                                text_content = data["text"]
                                total_chunks = data.get("total_chunks", "?")
                                
                                print(f"📝 片段 {chunk_index + 1}/{total_chunks}: {text_content[:50]}...")
                        
                        except json.JSONDecodeError as e:
                            print(f"JSON解析错误: {e} - {line_str}")
                        except Exception as e:
                            print(f"处理数据错误: {e}")
                
                print(f"\n总计接收 {chunk_count} 个音频片段")
                return audio_chunks
                
        except Exception as e:
            print(f"流式请求失败: {e}")
            return None
    
    async def batch_tts(self, text, audio_paths, **kwargs):
        """批量TTS"""
        request_data = {
            "text": text,
            "audio_paths": audio_paths,
            "seed": kwargs.get("seed", 8),
            "chunk_size": kwargs.get("chunk_size", 100),
            "sentence_split": kwargs.get("sentence_split", True)
        }
        
        print(f"发送批量TTS请求...")
        
        try:
            async with self.session.post(
                f"{self.server_url}/tts_batch",
                json=request_data
            ) as response:
                
                if response.status != 200:
                    error_text = await response.text()
                    print(f"批量请求失败: {response.status} - {error_text}")
                    return None
                
                # 保存完整音频
                audio_bytes = await response.read()
                output_file = "output_batch.wav"
                
                with open(output_file, "wb") as f:
                    f.write(audio_bytes)
                
                print(f"✓ 批量音频保存到: {output_file} ({len(audio_bytes)} 字节)")
                return output_file
                
        except Exception as e:
            print(f"批量请求失败: {e}")
            return None

async def main():
    """主函数"""
    # 测试文本
    test_text = """
    "十四五"以来，作为我国对外开放的重要平台，综合保税区、保税物流园区等海关特殊监管区域以不到两万分之一的国土面积，贡献了全国五分之一的进出口总值。
    河南、四川、重庆等"不沿边、不靠海"的中西部省市，依托海关特殊监管区域，外贸增速迅猛，特殊区域进出口值占到本地外贸的一半以上。
    2024年，海关特殊监管区域进出口值较2020年增长超三成。
    """.strip()
    
    # 音频样本路径（需要根据实际情况修改）
    audio_paths = [
        "/opt/notebook/data/TTS/index-tts-vllm-master/tests/sample_prompt.wav"
    ]
    
    # 如果样本文件不存在，尝试使用本地路径
    current_dir = Path(__file__).parent
    local_sample = current_dir / "sample_prompt.wav"
    if local_sample.exists():
        audio_paths = [str(local_sample)]
    
    print("🚀 开始测试流式TTS服务")
    print("=" * 60)
    
    async with StreamingTTSClient() as client:
        # 健康检查
        print("1. 健康检查...")
        health_ok = await client.test_health()
        if not health_ok:
            print("❌ 服务器不可用，请检查服务是否启动")
            return
        
        print("✅ 服务器健康检查通过\n")
        
        # 测试流式TTS
        print("2. 测试流式TTS...")
        start_time = time.time()
        
        audio_chunks = await client.stream_tts(
            text=test_text,
            audio_paths=audio_paths,
            chunk_size=80,
            sentence_split=True
        )
        
        stream_time = time.time() - start_time
        print(f"流式处理耗时: {stream_time:.2f} 秒\n")
        
        # 测试批量TTS
        print("3. 测试批量TTS...")
        start_time = time.time()
        
        batch_result = await client.batch_tts(
            text=test_text,
            audio_paths=audio_paths,
            chunk_size=80,
            sentence_split=True
        )
        
        batch_time = time.time() - start_time
        print(f"批量处理耗时: {batch_time:.2f} 秒\n")
        
        # 总结
        print("📊 测试总结:")
        print(f"- 流式处理: {stream_time:.2f}s, 生成 {len(audio_chunks) if audio_chunks else 0} 个片段")
        print(f"- 批量处理: {batch_time:.2f}s, 生成 {'1个完整文件' if batch_result else '失败'}")
        
        if audio_chunks:
            total_size = sum(chunk.get("size", 0) for chunk in audio_chunks)
            print(f"- 流式音频总大小: {total_size / 1024:.1f} KB")
        
        print("\n🎵 可以使用音频播放器播放生成的wav文件")

if __name__ == "__main__":
    asyncio.run(main())
