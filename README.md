# RT60 混响时间测量 API

基于 FastAPI + NumPy 的本地服务：上传单声道 16 位 PCM WAV 房间脉冲响应录音，返回 RT60 混响时间及各关键时刻。

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

## 测量算法

1. 以录音**绝对峰值**位置为测量起点。
2. 用**末尾 20% 样本**估计噪声均方功率。
3. 从峰值起按 **20 ms 帧**计算原始 RMS：首次出现**连续三帧**均不高于**噪声 RMS 两倍**时，以这三帧中首帧的起点为**有效尾端**，否则取录音终点；**噪声为零时**直接取录音终点。
4. 对有效区间逐样本计算 `max(样本平方 − 噪声功率, 0)`，**反向累积**（Schroeder 积分）并以**起点能量归一化**，得到分贝衰减曲线。
5. 对首次越过 **-5 dB** 和 **-25 dB** 的位置在相邻采样间作**线性插值**，`RT60 = 3 × (t₋₂₅dB − t₋₅dB)`。
6. 以下情况判定为无法测量：峰值 ≤ 4 倍噪声 RMS；有效区间 < 100 ms；有效区间能量为零；任一越线点缺失；-25 dB 越线点距有效尾端 < 20 ms。

## 项目结构

```
app/
  main.py      # FastAPI 入口：上传校验（422）与响应
  rt60.py      # 核心测量算法（NumPy）
tests/
  test_api.py  # 接口测试（约束校验、响应格式）
  test_rt60.py # 算法测试（合成信号验证各分支）
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
