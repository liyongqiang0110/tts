#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
流式TTS客户端 - 将长文本切分后流式处理并播放
"""

import asyncio
import aiohttp
import json
import time
import re
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path
import base64

class TextSplitter:
    """文本切分器"""
    
    @staticmethod
    def split_by_sentence(text: str, max_length: int = 100) -> List[str]:
        """按句子切分文本"""
        # 中文句子结束符
        sentence_endings = r'[。！？；\n]+'
        
        # 先按句子切分
        sentences = re.split(sentence_endings, text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # 如果单个句子就超过最大长度，需要进一步切分
            if len(sentence) > max_length:
                # 先保存当前chunk
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                
                # 切分长句子
                long_chunks = TextSplitter.split_by_length(sentence, max_length)
                chunks.extend(long_chunks)
            else:
                # 检查加入当前句子后是否超长
                if len(current_chunk) + len(sentence) + 1 <= max_length:
                    current_chunk += sentence
                else:
                    # 保存当前chunk并开始新的
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence
        
        # 保存最后的chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return [chunk for chunk in chunks if chunk]
    
    @staticmethod
    def split_by_length(text: str, max_length: int = 100) -> List[str]:
        """按固定长度切分文本"""
        chunks = []
        for i in range(0, len(text), max_length):
            chunk = text[i:i + max_length].strip()
            if chunk:
                chunks.append(chunk)
        return chunks

class StreamingTTSClient:
    """流式TTS客户端"""
    
    def __init__(self, server_url: str = "http://152.136.168.63:11996"):
        self.server_url = server_url.rstrip('/')
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def process_text_streaming(
        self,
        text: str,
        audio_paths: List[str],
        chunk_size: int = 100,
        sentence_split: bool = True,
        on_chunk_start: Optional[Callable] = None,
        on_chunk_complete: Optional[Callable] = None,
        on_progress: Optional[Callable] = None,
        on_error: Optional[Callable] = None
    ) -> List[Dict[str, Any]]:
        """
        流式处理长文本TTS
        
        Args:
            text: 要处理的文本
            audio_paths: 音频样本路径列表
            chunk_size: 文本切分大小
            sentence_split: 是否按句子切分
            on_chunk_start: 开始处理chunk的回调
            on_chunk_complete: 完成处理chunk的回调 
            on_progress: 进度更新回调
            on_error: 错误处理回调
        
        Returns:
            包含音频数据的字典列表
        """
        
        # 1. 文本切分
        print(f"📝 开始切分文本 (长度: {len(text)} 字符)")
        
        if sentence_split:
            chunks = TextSplitter.split_by_sentence(text, chunk_size)
        else:
            chunks = TextSplitter.split_by_length(text, chunk_size)
        
        print(f"✂️ 文本已切分为 {len(chunks)} 个片段")
        for i, chunk in enumerate(chunks):
            print(f"  片段 {i+1}: {chunk[:50]}{'...' if len(chunk) > 50 else ''}")
        
        # 2. 流式处理每个片段
        results = []
        total_chunks = len(chunks)
        
        for i, chunk_text in enumerate(chunks):
            try:
                # 触发开始处理回调
                if on_chunk_start:
                    await on_chunk_start(i, chunk_text, total_chunks)
                
                print(f"\n🎵 处理片段 {i+1}/{total_chunks}: {chunk_text[:50]}...")
                
                # 发送TTS请求
                start_time = time.time()
                audio_data = await self._request_tts(chunk_text, audio_paths)
                end_time = time.time()
                
                if audio_data:
                    result = {
                        'chunk_index': i,
                        'text': chunk_text,
                        'audio_data': audio_data,
                        'audio_size': len(audio_data),
                        'processing_time': end_time - start_time,
                        'timestamp': time.time()
                    }
                    results.append(result)
                    
                    print(f"✅ 片段 {i+1} 完成 ({len(audio_data)} 字节, {end_time - start_time:.2f}s)")
                    
                    # 触发完成回调
                    if on_chunk_complete:
                        await on_chunk_complete(i, result)
                else:
                    print(f"❌ 片段 {i+1} 处理失败")
                    if on_error:
                        await on_error(i, f"Failed to process chunk {i+1}")
                
                # 触发进度回调
                if on_progress:
                    await on_progress(i + 1, total_chunks, (i + 1) / total_chunks * 100)
                
                # 添加小延迟避免服务器压力过大
                await asyncio.sleep(0.1)
                
            except Exception as e:
                print(f"❌ 处理片段 {i+1} 时出错: {e}")
                if on_error:
                    await on_error(i, str(e))
                continue
        
        print(f"\n🎉 流式处理完成！成功生成 {len(results)} 个音频片段")
        return results
    
    async def _request_tts(self, text: str, audio_paths: List[str]) -> Optional[bytes]:
        """发送TTS请求"""
        if not self.session:
            raise RuntimeError("Session not initialized")
        
        data = {
            "text": text,
            "audio_paths": audio_paths
        }
        
        try:
            async with self.session.post(
                f"{self.server_url}/tts_url",
                json=data,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    return await response.read()
                else:
                    error_text = await response.text()
                    print(f"TTS请求失败: {response.status} - {error_text}")
                    return None
                    
        except asyncio.TimeoutError:
            print("TTS请求超时")
            return None
        except Exception as e:
            print(f"TTS请求异常: {e}")
            return None

class AudioPlayer:
    """音频播放器"""
    
    def __init__(self):
        self.audio_queue = []
        self.playing = False
    
    def add_audio(self, audio_data: bytes, chunk_index: int):
        """添加音频到播放队列"""
        # 保存音频文件
        filename = f"chunk_{chunk_index:03d}.wav"
        with open(filename, "wb") as f:
            f.write(audio_data)
        
        self.audio_queue.append({
            'index': chunk_index,
            'filename': filename,
            'data': audio_data
        })
        
        print(f"🎵 音频片段 {chunk_index} 已保存: {filename}")
    
    async def play_sequential(self):
        """顺序播放音频"""
        print(f"🔊 开始播放 {len(self.audio_queue)} 个音频片段")
        
        # 按索引排序
        sorted_audio = sorted(self.audio_queue, key=lambda x: x['index'])
        
        for audio_info in sorted_audio:
            print(f"播放: {audio_info['filename']}")
            # 这里可以集成实际的音频播放库
            # 例如: pygame, pydub, sounddevice等
            await asyncio.sleep(1)  # 模拟播放时间
        
        print("🎉 播放完成")

async def main():
    """主函数 - 演示流式TTS客户端"""
    
    # 测试文本
    test_text = """
    "十四五"以来，作为我国对外开放的重要平台，综合保税区、保税物流园区等海关特殊监管区域以不到两万分之一的国土面积，贡献了全国五分之一的进出口总值。
    河南、四川、重庆等"不沿边、不靠海"的中西部省市，依托海关特殊监管区域，外贸增速迅猛，特殊区域进出口值占到本地外贸的一半以上。
    2024年，海关特殊监管区域进出口值较2020年增长超三成。
    """.strip()
    
    # 音频样本路径
    audio_paths = ["/opt/notebook/data/TTS/index-tts-vllm-master/tests/sample_prompt.wav"]
    
    print("🚀 流式TTS客户端演示")
    print("=" * 60)
    
    # 创建音频播放器
    player = AudioPlayer()
    
    # 定义回调函数
    async def on_chunk_start(index, text, total):
        print(f"🎬 开始处理片段 {index+1}/{total}")
    
    async def on_chunk_complete(index, result):
        # 添加到播放队列
        player.add_audio(result['audio_data'], result['chunk_index'])
    
    async def on_progress(current, total, percentage):
        print(f"📊 进度: {current}/{total} ({percentage:.1f}%)")
    
    async def on_error(index, error):
        print(f"❌ 错误 - 片段 {index}: {error}")
    
    # 创建客户端并处理
    async with StreamingTTSClient("http://152.136.168.63:11996") as client:
        start_time = time.time()
        
        results = await client.process_text_streaming(
            text=test_text,
            audio_paths=audio_paths,
            chunk_size=80,
            sentence_split=True,
            on_chunk_start=on_chunk_start,
            on_chunk_complete=on_chunk_complete,
            on_progress=on_progress,
            on_error=on_error
        )
        
        total_time = time.time() - start_time
        
        print(f"\n📈 处理统计:")
        print(f"- 总耗时: {total_time:.2f} 秒")
        print(f"- 成功片段: {len(results)}")
        print(f"- 平均每片段: {total_time/len(results):.2f} 秒")
        
        if results:
            total_audio_size = sum(r['audio_size'] for r in results)
            print(f"- 音频总大小: {total_audio_size / 1024:.1f} KB")
        
        # 播放音频
        print("\n🎵 开始播放音频...")
        await player.play_sequential()

if __name__ == "__main__":
    asyncio.run(main())
