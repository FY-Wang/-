# 羊了个羊：星球 自动通关 Agent（YOLO/OpenCV + 窗口抓图 + Lookahead）

本项目提供可直接扩展到实机的自动化框架：

- **VisionParser 接 YOLO/OpenCV 输出**
- **窗口句柄 / ADB 抓图**
- **Greedy + Beam Search + MCTS lookahead**
- **一键运行入口 `main.py`**

## 1. 代码结构

- `agent.py`
  - 规则状态：`Tile`、`GameState`
  - 策略：`GreedyAgent`、`BeamSearchAgent`、`MCTSAgent`
  - 视觉：`Detection`、`OpenCVYOLODetector`、`OpenCVYOLOVisionParser`
  - 抓图源：`MSSScreenSource`、`Win32WindowSource`、`AdbSource`
  - 点击器：`PyAutoGuiClicker`、`AdbClicker`
  - 联动器：`AutoPlayer`
- `main.py`
  - 一键启动（demo/live）
- `sample_frame.json`
  - demo 模式示例输入

## 2. 安装依赖（按需）

最小测试运行（仅单元测试）不需要 YOLO/GUI 依赖。

实机运行建议安装：

```bash
pip install ultralytics opencv-python mss pyautogui
```

Windows 窗口句柄抓图还需要：

```bash
pip install pywin32
```

ADB 模式需要系统安装 adb。

## 3. 一键运行

### 3.1 Demo（不连接设备，快速验证策略链路）

```bash
python main.py --mode demo --strategy beam --frame-json sample_frame.json --steps 5
```

### 3.2 Live：桌面屏幕抓图 + YOLO + 鼠标点击

```bash
python main.py --mode live --source screen --strategy beam --weights yolo_tiles.pt --steps 50
```

### 3.3 Live：窗口句柄抓图 + YOLO + 鼠标点击（Windows）

```bash
python main.py --mode live --source window --window-hwnd 123456 --strategy mcts --weights yolo_tiles.pt
```

### 3.4 Live：ADB 抓图 + ADB 点击（Android/scrcpy 场景）

```bash
python main.py --mode live --source adb --adb-serial emulator-5554 --strategy beam --weights yolo_tiles.pt
```

## 4. VisionParser 如何接 YOLO 输出

`OpenCVYOLODetector` 已封装 ultralytics YOLO：

1. `frame_source.capture()` 抓到图像（numpy array）
2. `detector.detect(frame)` 输出 `Detection(label, bbox, confidence, layer)`
3. `OpenCVYOLOVisionParser` 将检测结果转换为 `ParsedFrame`
4. `AutoPlayer.step()` 自动执行：解析 -> 决策 -> 点击

## 5. Lookahead 策略说明

- `GreedyAgent`: 即时收益（凑三连 / 凑对子 / 解锁下层）
- `BeamSearchAgent`: 在 greedy 评分上做多步深度搜索
- `MCTSAgent`: 多次随机 rollout 估计动作期望价值

可通过 `--strategy greedy|beam|mcts` 一键切换。
