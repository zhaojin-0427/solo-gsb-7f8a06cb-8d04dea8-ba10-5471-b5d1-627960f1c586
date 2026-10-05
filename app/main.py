"""FastAPI 应用入口：POST /rt60 上传单声道 16 位 PCM WAV，返回 RT60 测量结果。"""

from __future__ import annotations

import io
import wave

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile

from .rt60 import (
    ALLOWED_SAMPLE_RATES,
    MAX_DURATION_S,
    MIN_DURATION_S,
    MeasurementError,
    measure_rt60,
)

# 10 秒 48 kHz 16 位单声道 PCM 约 960 KB，留足余量
MAX_UPLOAD_BYTES = 2_000_000

app = FastAPI(
    title="RT60 混响时间测量",
    version="1.0.0",
    description="上传单声道 16 位 PCM WAV（16 或 48 kHz，1～10 秒），测量房间脉冲响应的 RT60。",
)


def _unprocessable(detail: str) -> HTTPException:
    return HTTPException(status_code=422, detail=detail)


def _load_mono_pcm16_wav(data: bytes) -> tuple[np.ndarray, int]:
    """解析并校验 WAV，返回 (int16 采样数组, 采样率)。格式或范围错误抛 422。"""
    try:
        with wave.open(io.BytesIO(data), "rb") as w:
            channels = w.getnchannels()
            sampwidth = w.getsampwidth()
            sample_rate = w.getframerate()
            n_frames = w.getnframes()
            raw = w.readframes(n_frames)
    except Exception:
        raise _unprocessable("无法解析的 WAV 文件")

    if channels != 1:
        raise _unprocessable("仅支持单声道 WAV")
    if sampwidth != 2:
        raise _unprocessable("仅支持 16 位 PCM WAV")
    if sample_rate not in ALLOWED_SAMPLE_RATES:
        raise _unprocessable("采样率仅支持 16000 或 48000 Hz")

    try:
        samples = np.frombuffer(raw, dtype="<i2")
    except ValueError:
        raise _unprocessable("WAV 数据损坏")

    duration = samples.size / sample_rate
    if not (MIN_DURATION_S <= duration <= MAX_DURATION_S):
        raise _unprocessable("时长需在 1～10 秒之间")
    return samples, sample_rate


@app.post("/rt60")
async def rt60(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        raise _unprocessable("上传内容为空")
    if len(data) > MAX_UPLOAD_BYTES:
        raise _unprocessable("文件过大")

    samples, sample_rate = _load_mono_pcm16_wav(data)

    try:
        result = measure_rt60(samples, sample_rate)
    except MeasurementError as exc:
        return {"measurable": False, "reason": str(exc)}

    return {"measurable": True, **result}
