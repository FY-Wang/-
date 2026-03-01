# 羊了个羊：星球 自动通关 Agent（快速模板匹配 + 窗口抓图 + Lookahead）

本项目提供可直接扩展到实机的自动化框架：

- **VisionParser 支持快速砖块模板匹配（默认）和 YOLO**
- **自动识别羊了个羊窗口 / ADB 抓图**
- **Greedy + Beam Search + MCTS lookahead**
- **一键运行入口 `main.py`**

## 1. 代码结构

- `agent.py`
  - 规则状态：`Tile`、`GameState`
  - 策略：`GreedyAgent`、`BeamSearchAgent`、`MCTSAgent`
  - 视觉：`Detection`、`OpenCVTemplateDetector`、`OpenCVYOLODetector`、`OpenCVYOLOVisionParser`
  - 抓图源：`MSSScreenSource`、`Win32WindowSource`、`AutoWin32WindowSource`、`AdbSource`
  - 点击器：`PyAutoGuiClicker`、`AdbClicker`
  - 联动器：`AutoPlayer`
- `main.py`
  - 一键启动（demo/live）
- `sample_frame.json`
  - demo 模式示例输入

## 2. 安装依赖（按需）

最小测试运行（仅单元测试）不需要 YOLO/GUI 依赖。

模板匹配路径（推荐，快速）：

```bash
pip install opencv-python mss pyautogui
```

YOLO 路径（可选）：

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

### 3.2 Live：桌面屏幕抓图 + 快速模板匹配 + 鼠标点击（默认 detector）

```bash
python main.py --mode live --source screen --strategy beam --detector template --templates-dir templates --steps 50
```

### 3.3 Live：自动识别“羊了个羊”窗口 + 快速模板匹配（Windows，推荐）

```bash
python main.py --mode live --source window-auto --window-keywords 羊了个羊 --strategy beam --detector template --templates-dir templates
```

### 3.4 Live：手动句柄抓图 + YOLO + 鼠标点击（Windows，可选）

```bash
python main.py --mode live --source window --window-hwnd 123456 --strategy mcts --weights yolo_tiles.pt
```

### 3.5 Live：ADB 抓图 + ADB 点击（Android/scrcpy 场景，模板匹配）

```bash
python main.py --mode live --source adb --adb-serial emulator-5554 --strategy beam --detector template --templates-dir templates
```

## 4. 快速模板匹配（默认）

`OpenCVTemplateDetector` 使用 `cv2.matchTemplate` 对 `templates/*.png` 做快速匹配，适合固定 UI 图标识别：

1. 准备模板目录（文件名即 tile type，例如 `carrot.png`）
2. 程序抓图后对每个模板做归一化匹配
3. 通过阈值与简单去重生成 `Detection`
4. `OpenCVYOLOVisionParser` 统一转成 `ParsedFrame` 并继续决策点击

如需切换 YOLO，可传 `--detector yolo --weights xxx.pt`。

## 5. Lookahead 策略说明

- `GreedyAgent`: 即时收益（凑三连 / 凑对子 / 解锁下层）
- `BeamSearchAgent`: 在 greedy 评分上做多步深度搜索
- `MCTSAgent`: 多次随机 rollout 估计动作期望价值

可通过 `--strategy greedy|beam|mcts` 一键切换。


## 6. 自动窗口发现

`AutoWin32WindowSource` 会枚举可见窗口并按标题关键字匹配（默认 `羊了个羊`），无需手动提供 HWND。

可通过 `--window-keywords` 传多个关键词（逗号分隔），例如：

```bash
python main.py --mode live --source window-auto --window-keywords "羊了个羊,微信"
```


## 7. 运行无输出/无点击排查

现在程序启动会输出 `[startup]` 与每步 `[demo]/[live]` 日志。若无点击可优先检查：

1. 模板目录是否有 `*.png`（程序会打印文件数，且为空时直接报错）
2. `--source window-auto` 是否匹配到目标窗口标题
3. 是否误把模板阈值设得过高（需要时可调整 `OpenCVTemplateDetector`）

示例：

```bash
python main.py --mode live --source window-auto --window-keywords 羊了个羊 --detector template --templates-dir templates --steps 10
```
