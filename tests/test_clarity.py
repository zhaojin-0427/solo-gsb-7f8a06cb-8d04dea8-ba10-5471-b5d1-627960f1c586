"""C50 核心算法测试：合成信号验证分频带测量与各失败分支。"""

import numpy as np
import pytest
from scipy.signal import butter, sosfiltfilt

from app.clarity import OCTAVE_CENTER_FREQS, measure_clarity

FULL_SCALE = 32767.0


def decaying_band_noise(fs: float, duration: float, t60: float,
                        amplitude: float = 0.5, seed: int = 0) -> np.ndarray:
    """指数包络宽带噪声（量化为 int16）：每带能量均按同一包络衰减。"""
    rng = np.random.default_rng(seed)
    n = int(round(fs * duration))
    t = np.arange(n) / fs
    carrier = rng.normal(0.0, 1.0, n)
    sig = amplitude * FULL_SCALE * 10.0 ** (-3.0 * t / t60) * carrier
    return np.round(np.clip(sig, -32768, 32767)).astype(np.int16).astype(np.float64)


def expected_c50(t60: float, tail_s: float) -> float:
    """能量按 10^(-6t/t60) 衰减、0～tail_s 积分时的理论 C50（dB）。"""
    early = 1.0 - 10.0 ** (-0.3 / t60)
    late = 10.0 ** (-0.3 / t60) - 10.0 ** (-6.0 * tail_s / t60)
    return 10.0 * np.log10(early / late)


def test_bands_ordered_and_fields():
    fs = 48000
    result = measure_clarity(decaying_band_noise(fs, 2.0, 0.5), fs)
    freqs = [b["frequency_hz"] for b in result["bands"]]
    assert freqs == [500, 1000, 2000, 4000]
    assert result["start_time"] >= 0.0
    for b in result["bands"]:
        assert b["measurable"] is True
        assert set(b) == {"measurable", "frequency_hz", "c50_db", "tail_time"}


def test_fast_decay_positive_c50():
    fs = 48000
    duration = 2.0
    samples = decaying_band_noise(fs, duration, t60=0.3, seed=1)
    result = measure_clarity(samples, fs)
    target = expected_c50(0.3, duration)  # 约 9.5 dB（早期主导）
    for b in result["bands"]:
        assert b["measurable"] is True
        assert b["c50_db"] == pytest.approx(target, abs=2.5)
        assert b["c50_db"] > 5.0


def test_slow_decay_negative_c50():
    fs = 48000
    samples = decaying_band_noise(fs, 3.0, t60=2.0, seed=2)
    result = measure_clarity(samples, fs)
    target = expected_c50(2.0, 3.0)  # 约 -3.8 dB（晚期主导）
    for b in result["bands"]:
        assert b["measurable"] is True
        assert b["c50_db"] == pytest.approx(target, abs=2.5)
        assert b["c50_db"] < 0.0


def test_zero_noise_tail_is_recording_end():
    """数字静音衰减：末 20% 滤波后仍为零，噪声功率为零，各带尾端取录音终点。"""
    fs = 48000
    duration = 2.0
    samples = decaying_band_noise(fs, duration, t60=0.3, seed=3)
    result = measure_clarity(samples, fs)
    assert result["start_time"] == pytest.approx(0.0, abs=1 / fs)
    for b in result["bands"]:
        assert b["measurable"] is True
        assert b["tail_time"] == pytest.approx(duration, abs=1 / fs)


def test_silence_all_bands_unmeasurable():
    fs = 48000
    result = measure_clarity(np.zeros(int(fs * 2.0)), fs)
    assert len(result["bands"]) == 4
    for b in result["bands"]:
        assert b["measurable"] is False
        assert "四倍噪声 RMS" in b["reason"]


def test_common_start_is_raw_peak_not_filtered_peak():
    """所有频带共用原始波形绝对峰值位置：0.3 s 小脉冲、1.5 s 大脉冲，
    起点必须是 1.5 s，滤波后峰值不得重新对齐。"""
    fs = 48000
    n = int(3.0 * fs)
    samples = np.zeros(n)
    samples[int(0.3 * fs)] = 10000
    samples[int(1.5 * fs)] = 30000          # 原始绝对峰值（共同起点）
    samples[int(1.62 * fs)] = 20000         # 晚期脉冲，保证晚期有能量、尾端靠后
    result = measure_clarity(samples, fs)
    assert result["start_time"] == pytest.approx(1.5, abs=1 / fs)
    for b in result["bands"]:
        assert b["measurable"] is True
        # 尾端扫描从 1.5 s 开始：不可能早于 1.5 s
        assert b["tail_time"] > 1.6


def test_split_boundary_sample_counts_as_late():
    """边界采样归晚期：按定义独立复算零相位滤波后信号在 50 ms 边界处的早 /
    晚期能量划分，须与服务返回的 C50 一致；且把边界点误归早期会使晚期能量
    归零而判不可测，因此可测本身即要求边界归晚期。"""
    fs = 48000
    start = int(0.2 * fs)
    n = int(2.0 * fs)
    samples = np.zeros(n)
    samples[start] = 20000
    boundary = start + int(round(0.050 * fs))
    samples[boundary] = 20000
    result = measure_clarity(samples, fs)
    for b in result["bands"]:
        assert b["measurable"] is True
        fc = b["frequency_hz"]
        sos = butter(4, [fc / np.sqrt(2), fc * np.sqrt(2)],
                     btype="bandpass", output="sos", fs=fs)
        y = sosfiltfilt(sos, samples)
        # 零噪声尾端取终点；按“边界采样归晚期”复算
        e = y[start:] ** 2
        early = float(np.sum(e[: boundary - start]))
        late = float(np.sum(e[boundary - start:]))
        assert early > 0.0 and late > 0.0
        expected = 10.0 * np.log10(early / late)
        assert b["c50_db"] == pytest.approx(expected, abs=1e-9)


def test_short_tail_after_split_is_band_unmeasurable():
    """起点冲激 + 全程恒定低带噪声：冲激后首个 20 ms 帧起即连续三帧安静，
    本带尾端=20 ms，不超过 50 ms 分界，判不可测。"""
    fs = 48000
    n = int(2.0 * fs)
    rng = np.random.default_rng(4)
    # 仅覆盖 500 Hz 带的低带平稳噪声（低通至 400 Hz，对 1 kHz 以上带泄漏可忽略）
    low_noise = rng.normal(0.0, 3000.0, n)
    low_noise = sosfiltfilt(
        butter(4, 400.0, btype="lowpass", output="sos", fs=fs), low_noise
    )
    samples = low_noise
    samples[0] = 30000
    result = measure_clarity(samples, fs)
    by_freq = {b["frequency_hz"]: b for b in result["bands"]}
    assert by_freq[500]["measurable"] is False
    assert "分界" in by_freq[500]["reason"]
    # 高带无平稳噪声：冲激振铃能否被三帧安静规则判成短尾取决于带内振铃长度，
    # 无论可测与否，都不应报低信噪比原因
    for fc in (1000, 2000, 4000):
        if not by_freq[fc]["measurable"]:
            assert "四倍噪声 RMS" not in by_freq[fc]["reason"]


def test_band_independent_failure():
    """仅 4 kHz 带末尾含强 4 kHz 纯音（噪声 RMS 抬高、峰值不超过 4 倍），
    该带不可测；其余频带照常测量。"""
    fs = 48000
    duration = 2.0
    n = int(duration * fs)
    samples = decaying_band_noise(fs, duration, t60=0.5, seed=5)
    tail_start = int(0.8 * n)
    tone_t = np.arange(n - tail_start) / fs
    samples[tail_start:] += 0.9 * FULL_SCALE * np.sin(2 * np.pi * 4000.0 * tone_t)
    result = measure_clarity(samples, fs)
    by_freq = {b["frequency_hz"]: b for b in result["bands"]}
    assert by_freq[4000]["measurable"] is False
    assert "四倍噪声 RMS" in by_freq[4000]["reason"]
    for fc in (500, 1000, 2000):
        assert by_freq[fc]["measurable"] is True
        assert isinstance(by_freq[fc]["c50_db"], float)


def test_16k_highest_band_within_nyquist():
    fs = 16000
    result = measure_clarity(decaying_band_noise(fs, 2.0, 0.3, seed=6), fs)
    assert [b["frequency_hz"] for b in result["bands"]] == list(OCTAVE_CENTER_FREQS)
    for b in result["bands"]:
        assert b["measurable"] is True
