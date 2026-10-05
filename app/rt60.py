"""RT60 混响时间测量核心算法（纯 NumPy 实现）。

算法流程：
1. 以录音绝对峰值位置为起点；
2. 用末尾 20% 样本估计噪声均方功率；
3. 从峰值起按 20 ms 帧计算原始 RMS，首次连续三帧均不高于噪声 RMS 两倍时，
   以这三帧中首帧的起点为有效尾端，否则取录音终点；噪声为零时直接取录音终点；
4. 对有效区间逐样本计算 max(样本平方 - 噪声功率, 0)，反向累积（Schroeder 积分），
   并以起点能量归一化得到分贝衰减曲线；
5. 对首次越过 -5 dB 与 -25 dB 的位置在相邻采样间作线性插值，
   RT60 = 3 × (t_-25dB - t_-5dB)。
"""

from __future__ import annotations

import numpy as np

ALLOWED_SAMPLE_RATES = (16000, 48000)
MIN_DURATION_S = 1.0
MAX_DURATION_S = 10.0

FRAME_S = 0.020          # 帧长 20 ms
NOISE_TAIL_RATIO = 0.2   # 末尾 20% 样本用于噪声估计
NOISE_FRAME_FACTOR = 2.0  # 帧 RMS 不高于噪声 RMS 的 2 倍视为安静
PEAK_NOISE_FACTOR = 4.0  # 峰值需高于噪声 RMS 的 4 倍
MIN_INTERVAL_S = 0.100   # 有效区间至少 100 ms
TAIL_MARGIN_S = 0.020    # -25 dB 越线点距有效尾端至少 20 ms
LEVEL_NEAR_DB = -5.0
LEVEL_FAR_DB = -25.0
_DB_FLOOR = 1e-12        # 能量归一化下限，避免 log10(0)


class MeasurementError(Exception):
    """录音本身有效，但信号条件不满足，无法测量 RT60。"""


def measure_rt60(samples: np.ndarray, sample_rate: int) -> dict:
    """测量 RT60，返回各关键时刻（秒，相对录音起点）与 RT60（秒）。

    无法测量时抛出 MeasurementError，消息为中文原因。
    """
    x = np.asarray(samples, dtype=np.float64)
    n = x.size
    if n == 0:
        raise MeasurementError("录音为空")

    # 1. 起点：绝对峰值位置
    peak_idx = int(np.argmax(np.abs(x)))
    peak = float(abs(x[peak_idx]))

    # 2. 噪声：末尾 20% 样本的均方功率
    n_noise = max(1, int(round(n * NOISE_TAIL_RATIO)))
    noise_seg = x[n - n_noise:]
    noise_power = float(np.mean(noise_seg * noise_seg))
    noise_rms = float(np.sqrt(noise_power))

    # 3. 峰值必须高于四倍噪声 RMS
    if not peak > PEAK_NOISE_FACTOR * noise_rms:
        raise MeasurementError("峰值不高于四倍噪声 RMS，信噪比不足")

    # 4. 有效尾端
    if noise_power == 0.0:
        tail_idx = n
    else:
        tail_idx = _find_tail(x, peak_idx, noise_rms, sample_rate)

    # 5. 有效区间不得短于 100 ms
    if (tail_idx - peak_idx) < MIN_INTERVAL_S * sample_rate:
        raise MeasurementError("有效区间短于 100 毫秒")

    # 6. 分贝衰减曲线：max(样本平方 - 噪声功率, 0) 反向累积，以起点能量归一化
    seg = x[peak_idx:tail_idx]
    e = np.maximum(seg * seg - noise_power, 0.0)
    energy = np.cumsum(e[::-1])[::-1]
    if energy[0] <= 0.0:
        raise MeasurementError("有效区间能量为零")
    decay_db = 10.0 * np.log10(np.maximum(energy / energy[0], _DB_FLOOR))

    # 7. -5 dB / -25 dB 首次越线时间（相邻采样线性插值）
    t5 = _crossing_time(decay_db, LEVEL_NEAR_DB, peak_idx, sample_rate)
    t25 = _crossing_time(decay_db, LEVEL_FAR_DB, peak_idx, sample_rate)
    if t5 is None or t25 is None:
        raise MeasurementError("衰减曲线未首次越过 -5 dB 或 -25 dB")

    tail_time = tail_idx / sample_rate
    if tail_time - t25 < TAIL_MARGIN_S:
        raise MeasurementError("-25 dB 越线点距有效尾端不足 20 毫秒")

    return {
        "peak_time": peak_idx / sample_rate,
        "tail_time": tail_time,
        "t5_time": t5,
        "t25_time": t25,
        "rt60": 3.0 * (t25 - t5),
    }


def _find_tail(x: np.ndarray, peak_idx: int, noise_rms: float, sample_rate: int) -> int:
    """从峰值起按 20 ms 帧扫描，返回有效尾端采样索引（不含）。"""
    n = x.size
    frame = int(round(FRAME_S * sample_rate))
    limit = NOISE_FRAME_FACTOR * noise_rms
    start = peak_idx
    # 需要连续三个完整帧都不高于阈值
    while start + 3 * frame <= n:
        quiet = True
        for k in range(3):
            seg = x[start + k * frame: start + (k + 1) * frame]
            rms = float(np.sqrt(np.mean(seg * seg)))
            if rms > limit:
                quiet = False
                break
        if quiet:
            return start
        start += frame
    return n


def _crossing_time(
    decay_db: np.ndarray, level_db: float, peak_idx: int, sample_rate: int
) -> float | None:
    """衰减曲线首次越过 level_db 的时间（秒，相对录音起点），相邻采样线性插值。"""
    below = np.nonzero(decay_db <= level_db)[0]
    if below.size == 0:
        return None
    i = int(below[0])
    if i == 0:
        # 起点能量归一化后曲线首点恒为 0 dB，理论上不会触发；防御性处理
        return peak_idx / sample_rate
    l0 = float(decay_db[i - 1])
    l1 = float(decay_db[i])
    frac = (l0 - level_db) / (l0 - l1)
    return (peak_idx + (i - 1) + frac) / sample_rate
