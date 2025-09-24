
from fastapi.security.http import HTTPBearer


import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import asyncio
import io
import traceback
import uuid
import logging
import threading
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, Request, Response, HTTPException, Depends, File, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn
import argparse
import json
import time
import numpy as np
import soundfile as sf
import re
import base64
from typing import List, AsyncGenerator, Optional
import aiofiles

# TTS imports
from indextts.infer_vllm import IndexTTS

# ASR imports
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
ASR_AVAILABLE = True

# 全局变量
tts = None
asr_model = None  # ASR 模型
API_KEY = None  # 全局API密钥
args = None  # 命令行参数
security: HTTPBearer = HTTPBearer(auto_error=False)  # API密钥验证
model_lock = threading.Lock()  # 模型推理锁，避免并发冲突

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('unified_service.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ServiceConfig:
    """统一服务配置类"""
    def __init__(self):
        # TTS 配置
        self.tts_model_dir = "/path/to/IndexTeam/Index-TTS"
        self.gpu_memory_utilization = 0.25
        
        # ASR 配置
        self.asr_model_dir = "/opt/model/SenseVoiceSmall"
        self.upload_dir = "./uploads"
        self.max_file_size = 100 * 1024 * 1024  # 100MB
        self.supported_formats = ['.wav', '.mp3', '.m4a', '.flac', '.aac']
        self.device = "cuda:0"
        self.vad_kwargs = {"max_single_segment_time": 30000}
        self.batch_size_s = 60
        self.merge_length_s = 15


config = ServiceConfig()


def ensure_upload_directory():
    """确保上传目录存在"""
    upload_path = Path(config.upload_dir)
    upload_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"上传目录已准备: {upload_path.absolute()}")


def validate_audio_file(filename: str, file_size: int) -> bool:
    """验证音频文件格式和大小"""
    if file_size > config.max_file_size:
        raise HTTPException(
            status_code=413,
            detail=f"文件大小超出限制，最大允许 {config.max_file_size / 1024 / 1024:.1f}MB"
        )
    
    file_ext = Path(filename).suffix.lower()
    if file_ext not in config.supported_formats:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的音频格式 {file_ext}，支持的格式: {', '.join(config.supported_formats)}"
        )
    return True


async def save_uploaded_file(upload_file: UploadFile) -> str:
    """异步保存上传的音频文件"""
    filename = upload_file.filename or "unknown_file"
    file_ext = Path(filename).suffix.lower()
    unique_filename = f"{uuid.uuid4().hex}_{int(time.time())}{file_ext}"
    file_path = Path(config.upload_dir) / unique_filename
    
    try:
        async with aiofiles.open(file_path, 'wb') as f:
            content = await upload_file.read()
            await f.write(content)
        
        logger.info(f"音频文件保存成功: {file_path}, 大小: {len(content)} bytes")
        return str(file_path)
    except Exception as e:
        logger.error(f"保存音频文件失败: {e}")
        raise HTTPException(status_code=500, detail=f"文件保存失败: {str(e)}")


async def transcribe_audio(file_path: str, language: str = "auto", use_itn: bool = True) -> dict:
    """异步音频转文本处理"""
    global asr_model, model_lock
    
    if not ASR_AVAILABLE:
        raise HTTPException(status_code=503, detail="ASR功能不可用，请安装FunASR")
    
    if asr_model is None:
        raise HTTPException(status_code=503, detail="ASR模型未初始化")
    
    start_time = time.time()
    
    try:
        loop = asyncio.get_event_loop()
        
        def run_inference():
            with model_lock:
                logger.debug(f"获取模型锁，开始推理: {file_path}")
                result = asr_model.generate(
                    input=file_path,
                    cache={},
                    language=language,
                    use_itn=use_itn,
                    batch_size_s=config.batch_size_s,
                    merge_vad=True,
                    merge_length_s=config.merge_length_s,
                )
                logger.debug(f"推理完成，释放模型锁: {file_path}")
                return result
        
        res = await loop.run_in_executor(None, run_inference)
        raw_text = res[0]["text"] if res and len(res) > 0 else ""
        processed_text = rich_transcription_postprocess(raw_text)
        
        processing_time = time.time() - start_time
        
        result = {
            "text": processed_text,
            "raw_text": raw_text,
            "language": language,
            "processing_time": round(processing_time, 3),
            "audio_duration": res[0].get("duration", 0) if res and len(res) > 0 else 0,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"音频转文本完成: {file_path}, 处理时间: {processing_time:.3f}s, 文本长度: {len(processed_text)}")
        return result
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"音频转文本失败: {file_path}, 错误: {e}, 处理时间: {processing_time:.3f}s")
        raise HTTPException(status_code=500, detail=f"音频转文本失败: {str(e)}")


class TextSplitter:
    """文本分割器，支持自定义标点符号分割"""
    
    @staticmethod
    def split_by_punctuation(text: str, custom_punctuation: Optional[str] = None) -> List[str]:
        """按标点符号分割文本
        
        Args:
            text: 要分割的文本
            custom_punctuation: 自定义标点符号字符串，如 "：，。！？" 表示遇到这些标点都会分割
        """
        # 默认标点符号：中文句号、感叹号、问号、分号、省略号、换行符等
        default_punctuation = r'[。！？；…\n]+'
        
        # 如果提供了自定义标点，使用自定义标点
        if custom_punctuation and custom_punctuation.strip():
            # 去除重复字符并转义特殊字符
            unique_punct = ''.join(dict.fromkeys(custom_punctuation.strip()))
            escaped_punct = re.escape(unique_punct)
            sentence_endings = f'[{escaped_punct}]+'
            print(f"使用自定义标点符号分割: {unique_punct} -> 正则表达式: {sentence_endings}")
        else:
            sentence_endings = default_punctuation
            print(f"使用默认标点符号分割: {sentence_endings}")
        
        # 按标点分割
        sentences = re.split(sentence_endings, text.strip())
        # 过滤空字符串并清理前后空格
        chunks = [s.strip() for s in sentences if s.strip()]
        
        print(f"文本按标点符号分割为 {len(chunks)} 个片段")
        return chunks
    

async def generate_audio_stream(
    text: str, 
    audio_paths: List[str], 
    custom_punctuation: Optional[str] = None,
    seed: int = 8
) -> AsyncGenerator[str, None]:
    """流式生成音频数据"""
    global tts
    
    # 分割文本（只按标点符号分割，不限制长度）
    chunks = TextSplitter.split_by_punctuation(text, custom_punctuation)
    total_chunks = len(chunks)
    
    print(f"文本已分割为 {total_chunks} 个片段")
    
    # 发送初始信息
    yield f"data: {json.dumps({'type': 'start', 'total_chunks': total_chunks, 'chunks': chunks}, ensure_ascii=False)}\n\n"
    
    # 逐个处理文本片段
    for i, chunk_text in enumerate(chunks):
        try:
            print(f"正在处理片段 {i+1}/{total_chunks}: {chunk_text[:50]}...")
            
            # 发送当前处理的片段信息
            yield f"data: {json.dumps({'type': 'chunk_start', 'chunk_index': i, 'text': chunk_text, 'total_chunks': total_chunks}, ensure_ascii=False)}\n\n"
            
            # 生成音频
            start_time = time.time()
            sr, wav = await tts.infer(audio_paths, chunk_text, seed=seed)
            end_time = time.time()
            
            # 将音频转换为WAV格式的字节数据
            with io.BytesIO() as wav_buffer:
                sf.write(wav_buffer, wav, sr, format='WAV')
                wav_bytes = wav_buffer.getvalue()
            
            # 编码为base64
            audio_b64 = base64.b64encode(wav_bytes).decode('utf-8')
            
            # 发送音频数据
            audio_data = {
                'type': 'audio',
                'chunk_index': i,
                'data': audio_b64,
                'processing_time': end_time - start_time,
                'size': len(wav_bytes)
            }
            
            yield f"data: {json.dumps(audio_data, ensure_ascii=False)}\n\n"
            print(f"片段 {i+1} 完成，大小: {len(wav_bytes)} 字节，耗时: {end_time - start_time:.2f}s")
            
            # 短暂延迟，避免服务器压力过大
            await asyncio.sleep(0.1)
            
        except Exception as e:
            error_data = {
                'type': 'error',
                'chunk_index': i,
                'error': str(e)
            }
            yield f"data: {json.dumps(error_data, ensure_ascii=False)}\n\n"
            print(f"处理片段 {i+1} 时出错: {e}")
    
    # 发送完成信号
    yield f"data: {json.dumps({'type': 'complete', 'total_chunks': total_chunks}, ensure_ascii=False)}\n\n"


def cleanup_old_files():
    """清理旧的音频文件"""
    try:
        upload_path = Path(config.upload_dir)
        if not upload_path.exists():
            return
        
        current_time = time.time()
        cleanup_count = 0
        
        for file_path in upload_path.glob("*"):
            if file_path.is_file():
                # 删除1小时前的文件
                if current_time - file_path.stat().st_mtime > 3600:
                    file_path.unlink()
                    cleanup_count += 1
        
        if cleanup_count > 0:
            logger.info(f"清理了 {cleanup_count} 个旧音频文件")
            
    except Exception as e:
        logger.error(f"清理旧文件失败: {e}")


def verify_api_key(request: Request, credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> bool:
    """验证API密钥"""
    global API_KEY
    
    # 如果没有设置API密钥，则不进行验证
    if API_KEY is None:
        return True
    
    # 从多个地方获取API密钥
    api_key = None
    
    # 1. 从Authorization头获取（Bearer token）
    if credentials and credentials.credentials:
        api_key = credentials.credentials
    
    # 2. 从Authorization头获取
    if not api_key:
        api_key = request.headers.get("Authorization")
    
    # 3. 从API-Key头获取
    if not api_key:
        api_key = request.headers.get("API-Key")
    
    # 4. 从查询参数获取
    if not api_key:
        api_key = request.query_params.get("api_key")
    
    # 验证API密钥
    if not api_key or api_key != API_KEY:
        raise HTTPException(
            status_code=401,
            detail={
                "error": "Invalid or missing API key",
                "message": "Please provide a valid API key via Authorization header (Bearer token), Authorization header, API-Key header, or api_key query parameter"
            }
        )
    
    return True

@asynccontextmanager
async def lifespan(app: FastAPI):
    global tts, asr_model, API_KEY, args
    
    logger.info("🚀 统一AI服务启动中...")
    
    # 设置API密钥
    API_KEY = args.api_key or os.getenv("AI_API_KEY") or os.getenv("TTS_API_KEY")
    if API_KEY:
        logger.info(f"✅ API密钥验证已启用，密钥长度: {len(API_KEY)} 字符")
    else:
        logger.warning("⚠️ 未设置API密钥，所有请求都将被允许访问")
    
    ensure_upload_directory()
    
    # 预加载TTS模型
    try:
        logger.info(f"🔄 开始加载TTS模型: {config.tts_model_dir}")
        tts_start_time = time.time()
        
        cfg_path = os.path.join(config.tts_model_dir, "config.yaml")
        tts = IndexTTS(model_dir=config.tts_model_dir, cfg_path=cfg_path, gpu_memory_utilization=config.gpu_memory_utilization)

        current_file_path = os.path.abspath(__file__)
        cur_dir = os.path.dirname(current_file_path)
        speaker_path = os.path.join(cur_dir, "assets/speaker.json")
        if os.path.exists(speaker_path):
            speaker_dict = json.load(open(speaker_path, 'r'))
            for speaker, audio_paths in speaker_dict.items():
                audio_paths_ = []
                for audio_path in audio_paths:
                    audio_paths_.append(os.path.join(cur_dir, audio_path))
                tts.registry_speaker(speaker, audio_paths_)
                
        tts_load_time = time.time() - tts_start_time
        logger.info(f"✅ TTS模型加载完成，耗时: {tts_load_time:.2f}秒")
        
    except Exception as e:
        logger.error(f"❌ TTS模型加载失败: {e}")
        raise RuntimeError(f"TTS模型加载失败: {e}")
    
    # 预加载ASR模型（如果可用）
    if ASR_AVAILABLE and args.asr_model_dir:
        logger.info(f"🔄 开始加载ASR模型: {args.asr_model_dir}")
        asr_start_time = time.time()
        
        asr_model = AutoModel(
            model=args.asr_model_dir,
            trust_remote_code=True,
            remote_code="./model.py",
            vad_model="fsmn-vad",
            vad_kwargs=config.vad_kwargs,
            device=config.device,
        )
        
        asr_load_time = time.time() - asr_start_time
        logger.info(f"✅ ASR模型加载完成，耗时: {asr_load_time:.2f}秒")
            
    else:
        if not ASR_AVAILABLE:
            logger.info("跳过ASR模型加载（FunASR未安装）")
        else:
            logger.info("跳过ASR模型加载（未设置ASR模型目录）")
    
    # 清理旧文件
    cleanup_old_files()
    
    logger.info("🎉 统一AI服务启动成功！")
    logger.info(f"   - TTS服务: {'✅ 可用' if tts else '❌ 不可用'}")
    logger.info(f"   - ASR服务: {'✅ 可用' if asr_model else '❌ 不可用'}")
    
    yield
    
    logger.info("🛑 统一AI服务关闭中...")
    # 清理临时文件
    cleanup_old_files()
    logger.info("👋 统一AI服务已关闭")

app = FastAPI(lifespan=lifespan, title="统一AI服务API", description="提供TTS和ASR功能的统一API服务", version="1.0.0")

# 添加CORS中间件配置,避免跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源，生产环境建议改为具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check(request: Request, _: bool = Depends(verify_api_key)):
    """健康检查接口"""
    try:
        global tts, asr_model
        
        tts_status = tts is not None
        asr_status = asr_model is not None
        
        overall_status = "healthy" if (tts_status or asr_status) else "unhealthy"
        status_code = 200 if overall_status == "healthy" else 503
        
        return JSONResponse(
            status_code=status_code,
            content={
                "status": overall_status,
                "message": "ASR&&TTS service is running",
                "services": {
                    "tts": "available" if tts_status else "unavailable",
                    "asr": "available" if asr_status else "unavailable"
                },
                "tts_model_dir": config.tts_model_dir,
                "asr_model_dir": config.asr_model_dir if ASR_AVAILABLE else "N/A",
                "device": config.device,
                "timestamp": datetime.now().isoformat()
            }
        )
    except Exception as ex:
        logger.error(f"健康检查失败: {ex}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(ex)
            }
        )


@app.post("/tts_url", responses={
    200: {"content": {"application/octet-stream": {}}},
    500: {"content": {"application/json": {}}}
})
async def tts_api_url(request: Request, _: bool = Depends(verify_api_key)):
    try:
        data = await request.json()
        text = data["text"]
        audio_paths = data["audio_paths"]
        seed = data.get("seed", 8)

        global tts
        sr, wav = await tts.infer(audio_paths, text, seed=seed)
        
        with io.BytesIO() as wav_buffer:
            sf.write(wav_buffer, wav, sr, format='WAV')
            wav_bytes = wav_buffer.getvalue()

        return Response(content=wav_bytes, media_type="audio/wav")
    
    except Exception as ex:
        tb_str = ''.join(traceback.format_exception(type(ex), ex, ex.__traceback__))
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(tb_str)
            }
        )


@app.post("/tts", responses={
    200: {"content": {"application/octet-stream": {}}},
    500: {"content": {"application/json": {}}}
})
async def tts_api(request: Request, _: bool = Depends(verify_api_key)):
    try:
        data = await request.json()
        text = data["text"]
        character = data["character"]

        global tts
        sr, wav = await tts.infer_with_ref_audio_embed(character, text)
        
        with io.BytesIO() as wav_buffer:
            sf.write(wav_buffer, wav, sr, format='WAV')
            wav_bytes = wav_buffer.getvalue()

        return Response(content=wav_bytes, media_type="audio/wav")
    
    except Exception as ex:
        tb_str = ''.join(traceback.format_exception(type(ex), ex, ex.__traceback__))
        print(tb_str)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(tb_str)
            }
        )



@app.get("/audio/voices")
async def tts_voices(request: Request, _: bool = Depends(verify_api_key)):
    """ additional function to provide the list of available voices, in the form of JSON """
    current_file_path = os.path.abspath(__file__)
    cur_dir = os.path.dirname(current_file_path)
    speaker_path = os.path.join(cur_dir, "assets/speaker.json")
    if os.path.exists(speaker_path):
        speaker_dict = json.load(open(speaker_path, 'r'))
        return speaker_dict
    else:
        return []



@app.post("/audio/speech", responses={
    200: {"content": {"application/octet-stream": {}}},
    500: {"content": {"application/json": {}}}
})
async def tts_api_openai(request: Request, _: bool = Depends(verify_api_key)):
    """ OpenAI competible API, see: https://api.openai.com/v1/audio/speech """
    try:
        data = await request.json()
        text = data["input"]
        character = data["voice"]
        #model param is omitted
        _model = data["model"]

        global tts
        sr, wav = await tts.infer_with_ref_audio_embed(character, text)
        
        with io.BytesIO() as wav_buffer:
            sf.write(wav_buffer, wav, sr, format='WAV')
            wav_bytes = wav_buffer.getvalue()

        return Response(content=wav_bytes, media_type="audio/wav")
    
    except Exception as ex:
        tb_str = ''.join(traceback.format_exception(type(ex), ex, ex.__traceback__))
        print(tb_str)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(tb_str)
            }
        )


@app.post("/tts_streaming")
async def tts_streaming_api(request: Request, _: bool = Depends(verify_api_key)):
    """流式TTS接口 - 实时生成音频片段"""
    try:
        data = await request.json()
        text = data["text"]
        audio_paths = data["audio_paths"]
        seed = data.get("seed", 8)
        custom_punctuation = data.get("custom_punctuation", None)

        print(f"开始流式TTS处理，文本长度: {len(text)} 字符")
        print(f"分割参数: custom_punctuation={custom_punctuation}")
        
        # 返回Server-Sent Events流
        return StreamingResponse(
            generate_audio_stream(text, audio_paths, custom_punctuation, seed),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*",
            }
        )
    
    except Exception as ex:
        tb_str = ''.join(traceback.format_exception(type(ex), ex, ex.__traceback__))
        print(f"流式TTS处理错误: {tb_str}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(tb_str)
            }
        )


# ===== ASR 接口 =====

@app.get("/config")
async def get_config(request: Request, _: bool = Depends(verify_api_key)):
    """获取服务配置信息"""
    return JSONResponse(
        status_code=200,
        content={
            "supported_formats": config.supported_formats,
            "max_file_size_mb": config.max_file_size / 1024 / 1024,
            "asr_model_dir": config.asr_model_dir,
            "tts_model_dir": config.tts_model_dir,
            "device": config.device,
            "upload_dir": config.upload_dir,
            "batch_size_s": config.batch_size_s,
            "merge_length_s": config.merge_length_s,
            "asr_available": ASR_AVAILABLE and asr_model is not None,
            "tts_available": tts is not None
        }
    )


@app.post("/transcribe")
async def transcribe_audio_endpoint(
    request: Request,
    _: bool = Depends(verify_api_key),
    audio_file: UploadFile = File(...),
    language: str = "auto",
    use_itn: bool = True
):
    """
    音频转文本接口
    
    Parameters:
    - audio_file: 音频文件 (支持 wav, mp3, m4a, flac, aac)
    - language: 语言 ("auto", "zh", "en", "yue", "ja", "ko", "nospeech")
    - use_itn: 是否使用逆文本标准化
    
    Returns:
    - text: 识别的文本
    - raw_text: 原始文本
    - processing_time: 处理时间(秒)
    - audio_duration: 音频时长(秒)
    """
    if not ASR_AVAILABLE:
        raise HTTPException(status_code=503, detail="ASR功能不可用，请安装FunASR")
    
    request_id = uuid.uuid4().hex[:8]
    filename = audio_file.filename or "unknown_file"
    file_size = audio_file.size or 0
    client_host = request.client.host if request.client else "unknown"
    logger.info(f"[{request_id}] 收到转录请求: {filename}, 大小: {file_size} bytes, 客户端: {client_host}")
    
    try:
        # 验证文件
        validate_audio_file(filename, file_size)
        
        # 保存文件
        file_path = await save_uploaded_file(audio_file)
        
        try:
            # 执行转录
            result = await transcribe_audio(file_path, language, use_itn)
            result["request_id"] = request_id
            result["filename"] = audio_file.filename
            
            logger.info(f"[{request_id}] 转录成功: {result['text'][:100]}...")
            
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "data": result
                }
            )
            
        finally:
            # 清理临时文件
            try:
                Path(file_path).unlink(missing_ok=True)
                logger.debug(f"[{request_id}] 临时文件已清理: {file_path}")
            except Exception as e:
                logger.warning(f"[{request_id}] 清理临时文件失败: {e}")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] 转录请求处理失败: {e}")
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")


@app.post("/transcribe_batch")
async def transcribe_batch_endpoint(
    request: Request,
    _: bool = Depends(verify_api_key),
    audio_files: List[UploadFile] = File(...),
    language: str = "auto",
    use_itn: bool = True
):
    """
    批量音频转文本接口
    
    Parameters:
    - audio_files: 音频文件列表
    - language: 语言
    - use_itn: 是否使用逆文本标准化
    
    Returns:
    - results: 转录结果列表
    """
    if not ASR_AVAILABLE:
        raise HTTPException(status_code=503, detail="ASR功能不可用，请安装FunASR")
    
    request_id = uuid.uuid4().hex[:8]
    client_host = request.client.host if request.client else "unknown"
    logger.info(f"[{request_id}] 收到批量转录请求: {len(audio_files)} 个文件, 客户端: {client_host}")
    
    if len(audio_files) > 10:  # 限制批量处理数量
        raise HTTPException(status_code=400, detail="批量处理最多支持10个文件")
    
    results = []
    saved_files = []
    
    try:
        # 验证并保存所有文件
        for i, audio_file in enumerate(audio_files):
            filename = audio_file.filename or f"unknown_file_{i}"
            file_size = audio_file.size or 0
            validate_audio_file(filename, file_size)
            file_path = await save_uploaded_file(audio_file)
            saved_files.append((file_path, audio_file.filename))
        
        # 顺序处理所有文件（避免模型并发冲突）
        for i, (file_path, filename) in enumerate(saved_files):
            try:
                logger.info(f"[{request_id}] 处理文件 {i+1}/{len(saved_files)}: {filename}")
                result = await transcribe_audio(file_path, language, use_itn)
                result["filename"] = filename
                result["index"] = i
                results.append({
                    "filename": filename,
                    "success": True,
                    "data": result,
                    "index": i
                })
                logger.info(f"[{request_id}] 文件 {filename} 转录成功")
                
            except Exception as e:
                logger.error(f"[{request_id}] 文件 {filename} 转录失败: {e}")
                results.append({
                    "filename": filename,
                    "success": False,
                    "error": str(e),
                    "index": i
                })
        
        logger.info(f"[{request_id}] 批量转录完成: {len(results)} 个结果，成功: {sum(1 for r in results if r['success'])}, 失败: {sum(1 for r in results if not r['success'])}")
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "request_id": request_id,
                "total_files": len(audio_files),
                "results": results
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] 批量转录请求处理失败: {e}")
        raise HTTPException(status_code=500, detail=f"批量处理失败: {str(e)}")
    
    finally:
        # 清理所有临时文件
        for file_path, _ in saved_files:
            try:
                Path(file_path).unlink(missing_ok=True)
            except Exception as e:
                logger.warning(f"[{request_id}] 清理临时文件失败: {e}")


@app.get("/stats")
async def get_stats(request: Request, _: bool = Depends(verify_api_key)):
    """获取服务统计信息"""
    try:
        upload_path = Path(config.upload_dir)
        temp_files_count = len(list(upload_path.glob("*"))) if upload_path.exists() else 0
        
        return JSONResponse(
            status_code=200,
            content={
                "temp_files_count": temp_files_count,
                "upload_dir": str(upload_path.absolute()),
                "asr_model_loaded": asr_model is not None,
                "tts_model_loaded": tts is not None,
                "api_key_enabled": API_KEY is not None,
                "asr_available": ASR_AVAILABLE,
                "timestamp": datetime.now().isoformat()
            }
        )
    except Exception as e:
        logger.error(f"获取统计信息失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取统计信息失败: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="统一AI服务 - 提供TTS和ASR功能")
    
    # 通用参数
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务器主机地址")
    parser.add_argument("--port", type=int, default=11996, help="服务器端口")
    parser.add_argument("--api_key", type=str, default=None, 
                       help="API密钥用于验证请求。也可通过环境变量AI_API_KEY或TTS_API_KEY设置")
    
    # TTS 参数
    parser.add_argument("--model_dir", type=str, default="/path/to/IndexTeam/Index-TTS", 
                       help="TTS模型目录")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.25, 
                       help="GPU内存利用率")
    
    # ASR 参数
    parser.add_argument("--asr_model_dir", type=str, default=None, 
                       help="ASR模型目录，如果不设置则不启用ASR功能")
    parser.add_argument("--device", type=str, default="cuda:0", 
                       help="设备 (cuda:0, cpu)")
    parser.add_argument("--upload_dir", type=str, default="./uploads", 
                       help="音频文件上传目录")
    parser.add_argument("--max_file_size", type=int, default=100, 
                       help="最大文件大小(MB)")
    parser.add_argument("--batch_size_s", type=int, default=60, 
                       help="ASR批处理时长(秒)")
    parser.add_argument("--merge_length_s", type=int, default=15, 
                       help="ASR合并长度(秒)")
    
    args = parser.parse_args()
    
    # 设置全局args变量
    globals()['args'] = args
    
    # 更新配置
    config.tts_model_dir = args.model_dir
    config.gpu_memory_utilization = args.gpu_memory_utilization
    
    if args.asr_model_dir:
        config.asr_model_dir = args.asr_model_dir
    config.device = args.device
    config.upload_dir = args.upload_dir
    config.max_file_size = args.max_file_size * 1024 * 1024
    config.batch_size_s = args.batch_size_s
    config.merge_length_s = args.merge_length_s
    
    logger.info(f"启动统一AI服务: {args.host}:{args.port}")
    logger.info(f"TTS模型目录: {config.tts_model_dir}")
    if args.asr_model_dir:
        logger.info(f"ASR模型目录: {config.asr_model_dir}")
        logger.info(f"设备: {config.device}")
        logger.info(f"上传目录: {config.upload_dir}")
        logger.info(f"最大文件大小: {config.max_file_size / 1024 / 1024:.1f}MB")
    else:
        logger.info("ASR功能未启用（未设置ASR模型目录）")

    uvicorn.run(app=app, host=args.host, port=args.port)
