"""API 测试：校验上传约束（422）与测量响应格式。"""

import io
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

FULL_SCALE = 32767.0


def make_wav(samples: np.ndarray, fs: int, channels: int = 1, sampwidth: int = 2) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(sampwidth)
        w.setframerate(fs)
        if sampwidth == 2:
            frames = np.clip(samples, -32768, 32767).astype("<i2").tobytes()
        else:
            frames = samples.astype(np.uint8).tobytes()
        w.writeframes(frames)
    return buf.getvalue()


def decaying_ir(fs: int, duration: float = 3.0, t60: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(0)
    n = int(fs * duration)
    t = np.arange(n) / fs
    signal = 0.9 * FULL_SCALE * 10.0 ** (-3.0 * t / t60)
    return signal + rng.normal(0.0, 1e-4 * FULL_SCALE, n)


def post_wav(data: bytes, filename: str = "ir.wav"):
    return client.post("/rt60", files={"file": (filename, data, "audio/wav")})


def test_valid_wav_returns_rt60():
    resp = post_wav(make_wav(decaying_ir(48000), 48000))
    assert resp.status_code == 200
    body = resp.json()
    assert body["measurable"] is True
    assert body["rt60"] == pytest.approx(1.0, abs=0.05)
    for key in ("peak_time", "tail_time", "t5_time", "t25_time", "rt60"):
        assert isinstance(body[key], float)
    assert body["t5_time"] < body["t25_time"] < body["tail_time"]


def test_valid_wav_16k():
    resp = post_wav(make_wav(decaying_ir(16000, duration=2.0, t60=0.5), 16000))
    assert resp.status_code == 200
    assert resp.json()["rt60"] == pytest.approx(0.5, abs=0.05)


def test_unmeasurable_returns_reason_without_rt60():
    resp = post_wav(make_wav(np.zeros(48000), 48000))
    assert resp.status_code == 200
    body = resp.json()
    assert body["measurable"] is False
    assert body["reason"]
    assert "rt60" not in body


def test_non_wav_returns_422():
    resp = post_wav(b"this is not a wav file")
    assert resp.status_code == 422


def test_stereo_returns_422():
    mono = decaying_ir(48000)
    stereo = np.repeat(mono, 2)  # 交错成双声道
    resp = post_wav(make_wav(stereo, 48000, channels=2))
    assert resp.status_code == 422


def test_8bit_returns_422():
    samples = np.full(48000, 128, dtype=np.uint8)
    resp = post_wav(make_wav(samples, 48000, sampwidth=1))
    assert resp.status_code == 422


@pytest.mark.parametrize("fs", [8000, 44100])
def test_unsupported_sample_rate_returns_422(fs):
    resp = post_wav(make_wav(decaying_ir(fs, duration=2.0), fs))
    assert resp.status_code == 422


@pytest.mark.parametrize("duration", [0.5, 10.5])
def test_duration_out_of_range_returns_422(duration):
    fs = 16000
    resp = post_wav(make_wav(decaying_ir(fs, duration=duration), fs))
    assert resp.status_code == 422


def test_missing_file_returns_422():
    resp = client.post("/rt60")
    assert resp.status_code == 422
