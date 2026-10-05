# 房间脉冲响应测量 API

基于 FastAPI + NumPy/SciPy 的本地服务：上传单声道 16 位 PCM WAV 房间脉冲响应录音，测量 RT60 混响时间与各倍频程带的 C50 语音清晰度。

## 快速启动

```bash
docker compose up --build
```

一条命令即可启动，本地访问地址：**http://localhost:8000**

- 交互式接口文档（可直接上传文件测试）：http://localhost:8000/docs
- 服务监听端口：`8000`（容器内外均为 8000，可在 `docker-compose.yml` 中修改映射）

不使用 Docker 时也可以本地运行：

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 接口说明

### `POST /rt60`

以 `multipart/form-data` 上传 WAV 文件，表单字段名为 `file`。

```bash
curl -X POST http://localhost:8000/rt60 -F "file=@impulse.wav"
```

**输入约束**（不满足时返回 `422`）：

| 约束 | 取值 |
| --- | --- |
| 格式 | WAV（RIFF），16 位 PCM |
| 声道 | 单声道 |
| 采样率 | 16000 或 48000 Hz |
| 时长 | 1～10 秒 |

**测量成功**（`200`）：

```json
{
  "measurable": true,
  "peak_time": 0.0,
  "tail_time": 1.52,
  "t5_time": 0.09999,
  "t25_time": 0.4999,
  "rt60": 1.1997
}
```

所有时间量单位均为**秒**，且为相对录音起点的绝对时间：

- `peak_time`：绝对峰值位置（测量起点）
- `tail_time`：有效尾端
- `t5_time` / `t25_time`：衰减曲线首次越过 -5 dB / -25 dB 的时间（相邻采样线性插值）
- `rt60`：`3 × (t25_time - t5_time)`

**无法测量**（`200`，不返回 `rt60`，`reason` 说明原因）：

```json
{
  "measurable": false,
  "reason": "峰值不高于四倍噪声 RMS，信噪比不足"
}
```

可能的 `reason`：

- `峰值不高于四倍噪声 RMS，信噪比不足`
- `有效区间短于 100 毫秒`
- `有效区间能量为零`
- `衰减曲线未首次越过 -5 dB 或 -25 dB`
- `-25 dB 越线点距有效尾端不足 20 毫秒`

**格式或范围错误**（`422`）：非 WAV 文件、非单声道、非 16 位 PCM、采样率不为 16/48 kHz、时长超出 1～10 秒、文件为空或过大、缺少上传字段。

### `POST /clarity`

以上传 WAV 的同样方式测量各倍频程带的 C50 语音清晰度，输入约束与 `POST /rt60` 完全相同（不满足时同样返回 `422`）。

```bash
curl -X POST http://localhost:8000/clarity -F "file=@impulse.wav"
```

**响应**（`200`）：按中心频率升序列出 500、1000、2000、4000 Hz 四个倍频程带，每带给出状态与 `c50`（dB）或不可测 `reason`：

```json
{
  "bands": [
    {"center_hz": 500, "measurable": true, "c50": 5.46},
    {"center_hz": 1000, "measurable": true, "c50": 4.27},
    {"center_hz": 2000, "measurable": false, "reason": "峰值不高于四倍噪声 RMS，信噪比不足"},
    {"center_hz": 4000, "measurable": true, "c50": 4.29}
  ]
}
```

- `center_hz`：倍频程带中心频率（Hz）
- `measurable`：该带是否可测
- `c50`：可测时返回，早期/晚期能量比 `10·log10(早期能量/晚期能量)`，单位 **dB**
- `reason`：不可测时返回，仅该带受影响，其余频带照常测量

可能的 `reason`：

- `峰值不高于四倍噪声 RMS，信噪比不足`
- `有效尾端不超过起点后 50 毫秒分界`
- `早期能量为零`
- `晚期能量为零`

## 测量算法

### RT60

1. 以录音**绝对峰值**位置为测量起点。
2. 用**末尾 20% 样本**估计噪声均方功率。
3. 从峰值起按 **20 ms 帧**计算原始 RMS：首次出现**连续三帧**均不高于**噪声 RMS 两倍**时，以这三帧中首帧的起点为**有效尾端**，否则取录音终点；**噪声为零时**直接取录音终点。
4. 对有效区间逐样本计算 `max(样本平方 − 噪声功率, 0)`，**反向累积**（Schroeder 积分）并以**起点能量归一化**，得到分贝衰减曲线。
5. 对首次越过 **-5 dB** 和 **-25 dB** 的位置在相邻采样间作**线性插值**，`RT60 = 3 × (t₋₂₅dB − t₋₅dB)`。
6. 以下情况判定为无法测量：峰值 ≤ 4 倍噪声 RMS；有效区间 < 100 ms；有效区间能量为零；任一越线点缺失；-25 dB 越线点距有效尾端 < 20 ms。

### C50 语音清晰度

1. 以**原始波形绝对峰值**位置为所有频带**共同的直达声起点**，不按滤波后峰值重新对齐。
2. 用**四阶 Butterworth SOS 零相位带通**（`scipy.signal.sosfiltfilt`）分出中心频率 **500/1000/2000/4000 Hz** 的倍频程带，上下截止频率分别为 **fc/√2** 与 **fc×√2**。
3. 每带用**滤波后末尾 20% 样本**估计噪声均方功率。
4. 从共同起点按 **20 ms 完整帧**扫描：首次出现**连续三帧** RMS 均不高于**噪声 RMS 两倍**时，以首帧起点为**该带尾端**，否则取录音终点；**噪声为零时**直接取录音终点。各带独立确定尾端，不共用宽带尾端。
5. 对起点至尾端逐样本计算 `max(样本平方 − 噪声功率, 0)`，以**起点后 50 ms** 为界分成早期与晚期能量（**边界采样归晚期**），`C50 = 10·log10(早期能量/晚期能量)`，单位 dB。
6. 以下情况**仅该带**判定为不可测（其余频带照常测量）：共同起点至录音终点的带内峰值 ≤ 4 倍噪声 RMS；尾端不超过早晚分界；早期或晚期能量为零。

## 项目结构

```
app/
  main.py      # FastAPI 入口：上传校验（422）与响应
  rt60.py      # RT60 核心测量算法（NumPy）
  clarity.py   # C50 清晰度测量：倍频程带分频与早晚能量比（SciPy + NumPy）
tests/
  test_api.py     # 接口测试（/rt60 与 /clarity 的约束校验、响应格式）
  test_rt60.py    # RT60 算法测试（合成信号验证各分支）
  test_clarity.py # C50 算法测试（合成信号验证各频带分支）
Dockerfile
docker-compose.yml
requirements.txt       # 运行依赖
requirements-dev.txt   # 测试依赖
```

## 运行测试

```bash
pip install -r requirements-dev.txt
pytest
```
