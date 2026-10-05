"""C50 语音清晰度测量：按倍频程带分频，比较早期与晚期能量（SciPy 带通 + NumPy）。

算法流程：
1. 以原始波形绝对峰值位置为所有频带共同的直达声起点（不按滤波后峰值重新对齐）；
2. 用四阶 Butterworth SOS 零相位带通（sosfiltfilt）分出中心频率
   500/1000/2000/4000 Hz 的倍频程带，上下截止频率为 fc/√2 与 fc×√2；
3. 每带用滤波后末尾 20% 样本估计噪声均方功率；
4. 从共同起点按 20 ms 完整帧扫描：首次连续三帧 RMS 均不高于噪声 RMS 两倍时，
   以首帧起点为该带尾端，否则取录音终点；噪声为零时直接取录音终点
   （各带独立确定尾端，不共用宽带尾端）；
5. 对起点至尾端逐样本计算 max(样本平方 − 噪声功率, 0)，以起点后 50 ms 为界分早期/晚期
   （边界采样归晚期），C50 = 10·log10(早期能量 / 晚期能量)，单位 dB；
6. 以下情况仅该带不可测（其余频带照常测量）：共同起点至录音终点的带内峰值
   ≤ 4 倍噪声 RMS；尾端不超过早晚分界；早期或晚期能量为零。
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt

from .rt60 import (
    NOISE_TAIL_RATIO,
    PEAK_NOISE_FACTOR,
    MeasurementError,
    find_tail,
)

CENTER_FREQUENCIES_HZ = (500, 1000, 2000, 4000)  # 倍频程带中心频率，升序
FILTER_ORDER = 4                 # 四阶 Butterworth
EARLY_LATE_SPLIT_S = 0.050       # 早期/晚期分界：起点后 50 ms
_SQRT2 = float(np.sqrt(2.0))


def measure_clarity(samples: np.ndarray, sample_rate: int) -> list[dict]:
    """按中心频率升序返回各倍频程带的 C50 测量结果。

    每带一个 dict：measurable 为 True 时含 c50（dB），否则含 reason（中文原因）。
    """
    x = np.asarray(samples, dtype=np.float64)
    n = x.size
    if n == 0:
        raise MeasurementError("录音为空")

    # 共同直达声起点：原始波形绝对峰值位置，不按滤波后峰值重新对齐
    start_idx = int(np.argmax(np.abs(x)))

    bands = []
    for fc in CENTER_FREQUENCIES_HZ:
        sos = butter(
            FILTER_ORDER,
            [fc / _SQRT2, fc * _SQRT2],
            btype="bandpass",
            fs=sample_rate,
            output="sos",
        )
        filtered = sosfiltfilt(sos, x)
        bands.append({"center_hz": fc, **_measure_band(filtered, start_idx, sample_rate)})
    return bands


def _measure_band(filtered: np.ndarray, start_idx: int, sample_rate: int) -> dict:
    """测量单个频带，返回 {"measurable": True, "c50": ...} 或带原因的不可测结果。"""
    n = filtered.size

    # 噪声：滤波后末尾 20% 样本的均方功率
    n_noise = max(1, int(round(n * NOISE_TAIL_RATIO)))
    noise_seg = filtered[n - n_noise:]
    noise_power = float(np.mean(noise_seg * noise_seg))
    noise_rms = float(np.sqrt(noise_power))

    # 共同起点至录音终点的带内峰值需高于四倍噪声 RMS
    band_peak = float(np.max(np.abs(filtered[start_idx:])))
    if not band_peak > PEAK_NOISE_FACTOR * noise_rms:
        return {"measurable": False, "reason": "峰值不高于四倍噪声 RMS，信噪比不足"}

    # 本带尾端（各带独立，不共用宽带尾端）；零噪声时取录音终点
    if noise_power == 0.0:
        tail_idx = n
    else:
        tail_idx = find_tail(filtered, start_idx, noise_rms, sample_rate)

    # 早期/晚期分界：起点后 50 ms，边界采样归晚期
    split_idx = start_idx + int(round(EARLY_LATE_SPLIT_S * sample_rate))
    if tail_idx <= split_idx:
        return {"measurable": False, "reason": "有效尾端不超过起点后 50 毫秒分界"}

    seg = filtered[start_idx:tail_idx]
    e = np.maximum(seg * seg - noise_power, 0.0)
    early_energy = float(np.sum(e[: split_idx - start_idx]))
    late_energy = float(np.sum(e[split_idx - start_idx:]))
    if early_energy <= 0.0:
        return {"measurable": False, "reason": "早期能量为零"}
    if late_energy <= 0.0:
        return {"measurable": False, "reason": "晚期能量为零"}

    return {"measurable": True, "c50": 10.0 * float(np.log10(early_energy / late_energy))}
