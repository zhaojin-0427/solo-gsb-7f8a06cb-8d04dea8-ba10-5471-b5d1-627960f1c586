"""核心算法测试：用合成信号验证 RT60 测量与各类失败分支。"""

import numpy as np
import pytest

from app.rt60 import MeasurementError, measure_rt60

FULL_SCALE = 32767.0


def decaying_ir(fs: float, duration: float, t60: float, noise_rms: float,
                amplitude: float = 0.9, seed: int = 0) -> np.ndarray:
    """指数衰减脉冲响应 + 加性噪声，幅值包络 A(t) = amplitude * 10^(-3t/t60)。"""
    rng = np.random.default_rng(seed)
    n = int(round(fs * duration))
    t = np.arange(n) / fs
    signal = amplitude * FULL_SCALE * 10.0 ** (-3.0 * t / t60)
    noise = rng.normal(0.0, noise_rms * FULL_SCALE, n)
    return signal + noise


def test_exponential_decay_recovers_rt60_48k():
    fs = 48000
    samples = decaying_ir(fs, duration=3.0, t60=1.0, noise_rms=1e-4)
    result = measure_rt60(samples, fs)
    assert result["rt60"] == pytest.approx(1.0, abs=0.05)
    assert result["peak_time"] == pytest.approx(0.0, abs=1 / fs)
    assert 0.0 < result["tail_time"] <= 3.0
    assert result["t5_time"] < result["t25_time"] < result["tail_time"]
    assert result["rt60"] == pytest.approx(3.0 * (result["t25_time"] - result["t5_time"]))


def test_exponential_decay_recovers_rt60_16k():
    fs = 16000
    samples = decaying_ir(fs, duration=2.0, t60=0.5, noise_rms=1e-4)
    result = measure_rt60(samples, fs)
    assert result["rt60"] == pytest.approx(0.5, abs=0.05)


def test_zero_noise_uses_recording_end_as_tail():
    fs = 48000
    n = int(fs * 2.0)
    t = np.arange(n) / fs
    # 量化为整数（模拟 16 位 PCM），尾部成为真正的数字静音
    samples = np.round(0.9 * FULL_SCALE * 10.0 ** (-3.0 * t / 0.5))
    result = measure_rt60(samples, fs)
    assert result["tail_time"] == pytest.approx(2.0)
    assert result["rt60"] == pytest.approx(0.5, abs=0.05)


def test_silence_is_unmeasurable():
    samples = np.zeros(48000)
    with pytest.raises(MeasurementError, match="信噪比不足"):
        measure_rt60(samples, 48000)


def test_stationary_noise_fails_interval_check():
    fs = 48000
    rng = np.random.default_rng(1)
    samples = rng.normal(0.0, 0.15 * FULL_SCALE, int(fs * 2.0))
    with pytest.raises(MeasurementError, match="有效区间短于 100 毫秒"):
        measure_rt60(samples, fs)


def test_too_fast_decay_fails_interval_check():
    fs = 48000
    samples = decaying_ir(fs, duration=2.0, t60=0.05, noise_rms=1e-4)
    with pytest.raises(MeasurementError, match="有效区间短于 100 毫秒"):
        measure_rt60(samples, fs)


def test_crossing_missing_when_decay_never_reaches_25db():
    """稀疏脉冲串 + 区间末端大样本：前 40 帧各含一个脉冲使尾端延伸到 0.8 s，
    但能量集中在少数样本上，Schroeder 曲线最大衰减不足 25 dB，-25 dB 越线点缺失。"""
    fs = 16000
    n = fs  # 1 秒
    rng = np.random.default_rng(3)
    samples = rng.normal(0.0, 0.5, n)          # 极低噪声底（噪声窗为末尾纯噪声段）
    samples[0] = 100.0                          # 峰值冲激（起点）
    samples[320:12800:320] = 45.0               # 前 40 帧各一个脉冲，保持帧 RMS 高于阈值
    samples[12799] = 50.0                       # 区间末端大样本，使末端剩余能量占比高
    with pytest.raises(MeasurementError, match="未首次越过"):
        measure_rt60(samples, fs)


def test_crossing_too_close_to_tail():
    """恒定电平短促信号突然结束：Schroeder 曲线在尾端附近才越过 -25 dB。"""
    fs = 48000
    rng = np.random.default_rng(2)
    n = int(fs * 2.0)
    burst_len = int(fs * 0.4)
    samples = rng.normal(0.0, 1e-4 * FULL_SCALE, n)
    samples[:burst_len] += rng.uniform(-0.5, 0.5, burst_len) * FULL_SCALE
    samples[int(fs * 0.05)] = 0.9 * FULL_SCALE  # 固定峰值位置，保证有效区间足够长
    with pytest.raises(MeasurementError, match="距有效尾端不足 20 毫秒"):
        measure_rt60(samples, fs)
