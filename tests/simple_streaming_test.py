#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的流式TTS测试脚本
"""

import asyncio
import time
from streaming_client import StreamingTTSClient

async def simple_test():
    """简单测试"""
    
    # 测试文本
    test_text = """
    "十四五"以来，作为我国对外开放的重要平台，综合保税区、保税物流园区等海关特殊监管区域以不到两万分之一的国土面积，贡献了全国五分之一的进出口总值。
    河南、四川、重庆等"不沿边、不靠海"的中西部省市，依托海关特殊监管区域，外贸增速迅猛，特殊区域进出口值占到本地外贸的一半以上。
    2024年，海关特殊监管区域进出口值较2020年增长超三成。
    """.strip()
    
    # 音频样本路径
    audio_paths = ["/opt/notebook/data/TTS/index-tts-vllm-master/tests/sample_prompt.wav"]
    
    print("🚀 开始简单流式TTS测试")
    print("=" * 50)
    
    async with StreamingTTSClient("http://152.136.168.63:11996") as client:
        start_time = time.time()
        
        results = await client.process_text_streaming(
            text=test_text,
            audio_paths=audio_paths,
            chunk_size=60,
            sentence_split=True
        )
        
        total_time = time.time() - start_time
        
        print(f"\n📊 测试结果:")
        print(f"- 总耗时: {total_time:.2f} 秒")
        print(f"- 成功片段: {len(results)}")
        if results:
            print(f"- 平均每片段: {total_time/len(results):.2f} 秒")
            total_size = sum(r['audio_size'] for r in results)
            print(f"- 音频总大小: {total_size / 1024:.1f} KB")
        
        print(f"\n生成的音频文件:")
        for i, result in enumerate(results):
            print(f"  chunk_{i:03d}.wav - {result['audio_size']} 字节")

if __name__ == "__main__":
    asyncio.run(simple_test())
