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


def decaying_band_ir(fs: int, duration: float = 3.0, t60: float = 0.5,
                     seed: int = 7) -> np.ndarray:
    """指数包络宽带噪声脉冲响应：四个倍频程带均有持续衰减能量。"""
    rng = np.random.default_rng(seed)
    n = int(fs * duration)
    t = np.arange(n) / fs
    return 0.5 * FULL_SCALE * 10.0 ** (-3.0 * t / t60) * rng.normal(0.0, 1.0, n) \
        + rng.normal(0.0, 1e-4 * FULL_SCALE, n)


def post_wav(data: bytes, filename: str = "ir.wav"):
    return client.post("/rt60", files={"file": (filename, data, "audio/wav")})


def post_clarity(data: bytes, filename: str = "ir.wav"):
    return client.post("/clarity", files={"file": (filename, data, "audio/wav")})


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


# ---------- POST /clarity ----------

def test_clarity_valid_wav_returns_bands():
    resp = post_clarity(make_wav(decaying_band_ir(48000), 48000))
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body["start_time"], float)
    bands = body["bands"]
    assert [b["frequency_hz"] for b in bands] == [500, 1000, 2000, 4000]
    for b in bands:
        assert b["measurable"] is True
        assert isinstance(b["c50_db"], float)
        assert isinstance(b["tail_time"], float)
        assert "reason" not in b


def test_clarity_silence_lists_per_band_reasons():
    resp = post_clarity(make_wav(np.zeros(48000), 48000))
    assert resp.status_code == 200
    bands = resp.json()["bands"]
    assert len(bands) == 4
    for b in bands:
        assert b["measurable"] is False
        assert b["reason"]
        assert "c50_db" not in b


def test_clarity_16k_supported():
    ir = decaying_band_ir(16000, duration=2.0, t60=0.4)
    resp = post_clarity(make_wav(ir, 16000))
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["bands"]) == 4
    assert all(b["measurable"] for b in body["bands"])


def test_clarity_non_wav_returns_422():
    resp = post_clarity(b"this is not a wav file")
    assert resp.status_code == 422


def test_clarity_stereo_returns_422():
    stereo = np.repeat(decaying_ir(48000), 2)
    resp = post_clarity(make_wav(stereo, 48000, channels=2))
    assert resp.status_code == 422


def test_clarity_8bit_returns_422():
    samples = np.full(48000, 128, dtype=np.uint8)
    resp = post_clarity(make_wav(samples, 48000, sampwidth=1))
    assert resp.status_code == 422


@pytest.mark.parametrize("fs", [8000, 44100])
def test_clarity_unsupported_sample_rate_returns_422(fs):
    resp = post_clarity(make_wav(decaying_ir(fs, duration=2.0), fs))
    assert resp.status_code == 422


@pytest.mark.parametrize("duration", [0.5, 10.5])
def test_clarity_duration_out_of_range_returns_422(duration):
    fs = 16000
    resp = post_clarity(make_wav(decaying_ir(fs, duration=duration), fs))
    assert resp.status_code == 422


def test_clarity_empty_and_oversize_returns_422():
    assert post_clarity(b"").status_code == 422
    big = make_wav(decaying_ir(48000, duration=10.0), 48000)
    # 伪造超过上传上限的负载（服务端只看字节数）
    resp = post_clarity(big + b"\x00" * 2_000_001)
    assert resp.status_code == 422


def test_clarity_missing_file_returns_422():
    resp = client.post("/clarity")
    assert resp.status_code == 422
