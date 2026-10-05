"""分频带语音清晰度 C50 测量核心算法（NumPy + SciPy）。

算法流程：
1. 以**原始波形**绝对峰值位置作为所有频带共同的直达声起点（不按滤波后峰值
   重新对齐）；
2. 对中心频率 500 / 1000 / 2000 / 4000 Hz 的倍频程带（上下截止频率分别为
   中心频率 /√2 与中心频率 ×√2），分别用 SciPy 四阶 Butterworth SOS 零相位
   带通滤波；
3. 每带用滤波后末尾 20% 样本估计噪声功率；从共同起点起按 20 ms 完整帧扫描，
   首次连续三帧 RMS 均不高于两倍噪声 RMS 时取首帧起点为本带尾端，否则取
   录音终点；噪声为零时同样取录音终点（各带独立判定，不共用宽带尾端）；
4. 每带对起点至尾端样本计算 max(样本平方 - 噪声功率, 0)，以起点后 50 ms
   为界分成早期 / 晚期能量（边界采样归晚期），C50 = 10·log10(早期/晚期) dB；
5. 带内峰值（共同起点至录音终点）不高于四倍噪声 RMS、尾端不超过 50 ms 分界，
   或早 / 晚期能量任一为零时，仅该带返回不可测原因，其余频带照常测量。
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

# 复用 /rt60 的 20 ms 完整帧尾端扫描逻辑：三帧连续 RMS ≤ 2 倍噪声 RMS，
# 首帧起点为尾端，扫不到则取录音终点。
from .rt60 import NOISE_TAIL_RATIO, _find_tail

OCTAVE_CENTER_FREQS = (500.0, 1000.0, 2000.0, 4000.0)
FILTER_ORDER = 4
PEAK_NOISE_FACTOR = 4.0   # 带内峰值需高于噪声 RMS 的 4 倍
EARLY_LATE_SPLIT_S = 0.050  # 早期 / 晚期能量分界：起点后 50 ms
_DB_FLOOR = 1e-12         # 能量比值下限，避免 log10(0)
# 输入恒为 16 位 PCM（LSB=1）：零相位滤波会在数字静音段留下远低于量化电平的
# 浮点尾数（约 1e-140 量级），噪声 RMS 低于该容差时按零噪声处理。
NOISE_RMS_EPS = 1e-12


def measure_clarity(samples: np.ndarray, sample_rate: int) -> dict:
    """测量四个倍频程带的 C50。

    返回 {"start_time": float, "bands": [每带结果, ...]}，bands 按中心频率
    升序排列；单带不可测时该带为 {"measurable": false, "reason": ...}，
    不影响其他频带。
    """
    x = np.asarray(samples, dtype=np.float64)
    n = x.size
    if n == 0:
        raise ValueError("录音为空")

    # 1. 共同起点：原始波形绝对峰值位置（滤波后不再重新对齐）
    start_idx = int(np.argmax(np.abs(x)))

    # 2. 早期 / 晚期能量分界采样（边界采样归晚期）
    split_idx = start_idx + int(round(EARLY_LATE_SPLIT_S * sample_rate))

    bands: list[dict] = []
    for fc in OCTAVE_CENTER_FREQS:
        bands.append(_measure_band(x, fc, sample_rate, start_idx, split_idx))

    return {"start_time": start_idx / sample_rate, "bands": bands}


def _measure_band(
    x: np.ndarray,
    center_hz: float,
    sample_rate: int,
    start_idx: int,
    split_idx: int,
) -> dict:
    """测量单个倍频程带；不可测时返回 measurable=False 及中文原因。"""
    n = x.size
    low = center_hz / np.sqrt(2.0)
    high = center_hz * np.sqrt(2.0)
    sos = butter(FILTER_ORDER, [low, high], btype="bandpass",
                 output="sos", fs=sample_rate)
    y = sosfiltfilt(sos, x)

    # 3. 末尾 20% 样本估计噪声功率（每带独立）
    n_noise = max(1, int(round(n * NOISE_TAIL_RATIO)))
    noise_seg = y[n - n_noise:]
    noise_power = float(np.mean(noise_seg * noise_seg))
    noise_rms = float(np.sqrt(noise_power))

    # 4. 带内峰值：共同起点至录音终点，需高于四倍噪声 RMS
    band_peak = float(np.max(np.abs(y[start_idx:])))
    if not band_peak > PEAK_NOISE_FACTOR * noise_rms:
        return _unmeasurable(center_hz, "带内峰值不高于四倍噪声 RMS，信噪比不足")

    # 5. 本带尾端：20 ms 完整帧扫描；零噪声或扫不到连续三帧时取录音终点
    if noise_rms <= NOISE_RMS_EPS:
        tail_idx = n
    else:
        tail_idx = _find_tail(y, start_idx, noise_rms, sample_rate)

    # 6. 尾端须越过 50 ms 分界（边界归晚期，晚期至少含一个样本）
    if tail_idx <= split_idx:
        return _unmeasurable(center_hz, "尾端不超过早期/晚期 50 毫秒分界")

    # 7. max(样本平方 - 噪声功率, 0) 后按 50 ms 分界分能量（边界归晚期）
    seg = y[start_idx:tail_idx]
    e = np.maximum(seg * seg - noise_power, 0.0)
    early = float(np.sum(e[: split_idx - start_idx]))
    late = float(np.sum(e[split_idx - start_idx:]))
    if early <= 0.0:
        return _unmeasurable(center_hz, "早期能量为零")
    if late <= 0.0:
        return _unmeasurable(center_hz, "晚期能量为零")

    c50 = 10.0 * float(np.log10(max(early / late, _DB_FLOOR)))
    return {
        "measurable": True,
        "frequency_hz": int(center_hz),
        "c50_db": c50,
        "tail_time": tail_idx / sample_rate,
    }


def _unmeasurable(center_hz: float, reason: str) -> dict:
    return {"measurable": False, "frequency_hz": int(center_hz), "reason": reason}
