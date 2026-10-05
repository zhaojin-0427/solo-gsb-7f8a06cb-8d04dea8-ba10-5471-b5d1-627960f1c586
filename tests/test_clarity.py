"""C50 清晰度算法测试：用合成信号验证各频带测量与各类失败分支。"""

import numpy as np
import pytest

from app.clarity import measure_clarity

FULL_SCALE = 32767.0


def decaying_noise_ir(fs: float, duration: float, t60: float,
                      noise_rms: float = 1e-4, seed: int = 0) -> np.ndarray:
    """衰减白噪声脉冲响应 + 加性噪声，幅值包络按 10^(-3t/t60) 衰减。"""
    rng = np.random.default_rng(seed)
    n = int(round(fs * duration))
    t = np.arange(n) / fs
    ir = rng.normal(0.0, 0.5 * FULL_SCALE, n) * 10.0 ** (-3.0 * t / t60)
    return ir + rng.normal(0.0, noise_rms * FULL_SCALE, n)


def band_of(bands: list[dict], center_hz: int) -> dict:
    return next(b for b in bands if b["center_hz"] == center_hz)


def test_bands_sorted_by_center_frequency():
    bands = measure_clarity(decaying_noise_ir(48000, 3.0, 0.5), 48000)
    assert [b["center_hz"] for b in bands] == [500, 1000, 2000, 4000]


def test_decaying_noise_all_bands_measurable_48k():
    # 能量按 10^(-6t/T60) 衰减时理论 C50 = 10log10((1-10^-0.6)/10^-0.6) ≈ 4.74 dB
    bands = measure_clarity(decaying_noise_ir(48000, 3.0, 0.5), 48000)
    for b in bands:
        assert b["measurable"] is True
        assert b["c50"] == pytest.approx(4.74, abs=1.5)


def test_decaying_noise_all_bands_measurable_16k():
    bands = measure_clarity(decaying_noise_ir(16000, 2.0, 0.5), 16000)
    for b in bands:
        assert b["measurable"] is True
        assert b["c50"] == pytest.approx(4.74, abs=1.5)


def test_tone_c50_matches_theory():
    """1000 Hz 衰减单音（T60=0.5）：本带 C50 精确匹配理论值；两倍频程外的
    4000 Hz 带只剩噪声，单独返回不可测，不影响本带测量。"""
    fs = 48000
    n = int(fs * 2.0)
    t = np.arange(n) / fs
    rng = np.random.default_rng(0)
    samples = (0.9 * FULL_SCALE * np.sin(2 * np.pi * 1000 * t) * 10.0 ** (-3.0 * t / 0.5)
               + rng.normal(0.0, 1e-4 * FULL_SCALE, n))
    bands = measure_clarity(samples, fs)
    assert band_of(bands, 1000)["c50"] == pytest.approx(4.74, abs=0.3)
    assert band_of(bands, 4000)["measurable"] is False


def test_common_start_uses_raw_peak_not_filtered():
    """共同起点由原始波形绝对峰值决定：1000 Hz 组分在 0.05～0.25 s，500 Hz 大组分
    在 0.3 s 起。从共同起点（≈0.3 s）看，1000 Hz 带只剩噪声，故不可测；若按滤波后
    峰值重新对齐则会落在 1000 Hz 组分上而变成可测。"""
    fs = 48000
    n = int(fs * 2.0)
    t = np.arange(n) / fs
    rng = np.random.default_rng(1)
    samples = rng.normal(0.0, 1e-2 * FULL_SCALE, n)
    m = (t >= 0.05) & (t < 0.25)
    samples[m] += 0.3 * FULL_SCALE * np.sin(2 * np.pi * 1000 * t[m])
    m = t >= 0.3
    samples[m] += (0.9 * FULL_SCALE * np.sin(2 * np.pi * 500 * t[m])
                   * 10.0 ** (-3.0 * (t[m] - 0.3) / 0.8))
    bands = measure_clarity(samples, fs)
    assert band_of(bands, 1000)["measurable"] is False
    # 500 Hz 带从共同起点正常测量，C50 符合 T60=0.8 的理论值 ≈ 1.39 dB
    assert band_of(bands, 500)["c50"] == pytest.approx(1.39, abs=0.3)


def test_per_band_tail_not_shared():
    """500 Hz 短促大爆发（5 ms）+ 2000 Hz 长衰减音：500 Hz 带自身尾端落在 50 ms
    分界前（不可测），2000 Hz 带尾端延伸到音结束（可测）——各带独立确定尾端。"""
    fs = 48000
    n = int(fs * 2.0)
    t = np.arange(n) / fs
    rng = np.random.default_rng(2)
    samples = rng.normal(0.0, 3e-2 * FULL_SCALE, n)
    m = (t >= 0.1) & (t < 0.105)
    samples[m] += 0.6 * FULL_SCALE * np.sin(2 * np.pi * 500 * t[m])
    samples += 0.3 * FULL_SCALE * np.sin(2 * np.pi * 2000 * t) * 10.0 ** (-3.0 * t / 1.0)
    bands = measure_clarity(samples, fs)
    band500 = band_of(bands, 500)
    assert band500["measurable"] is False
    assert "尾端" in band500["reason"]
    assert band_of(bands, 2000)["measurable"] is True


def test_zero_noise_uses_recording_end():
    """单个冲激 + 长静音：滤波后末尾 20% 的均方功率精确为零（IIR 响应下溢），
    零噪声时取录音终点为尾端，各带仍可测。"""
    fs = 16000
    samples = np.zeros(int(fs * 2.0))
    samples[0] = 0.9 * FULL_SCALE
    bands = measure_clarity(samples, fs)
    for b in bands:
        assert b["measurable"] is True
        assert b["c50"] > 60.0  # 冲激响应能量集中在早期


def test_silence_all_bands_unmeasurable():
    bands = measure_clarity(np.zeros(48000), 48000)
    for b in bands:
        assert b["measurable"] is False
        assert "信噪比不足" in b["reason"]
