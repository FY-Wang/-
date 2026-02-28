# Triple-Match Stack Agent + Vision Autoplay Skeleton

这个仓库提供了一个可扩展的《羊了个羊：星球》自动通关脚手架：

1. 规则状态建模（堆叠、遮挡、7槽、三连自动消除）
2. 策略决策（Greedy）
3. 视觉解析 + 自动点击执行接口（截图/窗口捕捉联动）

## 核心模块

- `Tile` / `GameState`
  - 维护 tiles、covers、可见性判定、shelf 与自动三消
- `GreedyAgent`
  - 优先立即三消
  - 其次补对子
  - 再考虑解锁下层牌
- `ParsedFrame` / `VisionParser` / `ClickController` / `AutoPlayer`
  - `VisionParser`: 负责“捕捉窗口 + 解析 tiles/covers”
  - `ClickController`: 负责点击屏幕坐标
  - `AutoPlayer.step()`: 每步自动“截图解析 -> 策略选牌 -> 执行点击”

## Quick Start (策略模拟)

```python
from agent import Tile, GameState, GreedyAgent, run_episode

tiles = {
    1: Tile(1, "carrot", 2, (0, 0)),
    2: Tile(2, "carrot", 1, (0, 0)),
    3: Tile(3, "carrot", 0, (0, 0)),
}

covers = {1: {2}, 2: {3}, 3: set()}
state = GameState(tiles=tiles, covers=covers, shelf_capacity=7)
print(run_episode(state, GreedyAgent()))
```

## Quick Start (视觉自动点击联动)

```python
from agent import ParsedFrame, ParsedTile, AutoPlayer, VisionParser, ClickController

class MyParser(VisionParser):
    def capture_and_parse(self) -> ParsedFrame:
        # 1) 截图窗口
        # 2) 检测 icon/type/layer/position
        # 3) 推断 covers 图
        return ParsedFrame(
            tiles=[
                ParsedTile(id=1, type="carrot", layer=1, position=(200, 300)),
                ParsedTile(id=2, type="carrot", layer=0, position=(200, 350)),
            ],
            covers={1: {2}, 2: set()},
        )

class MyClicker(ClickController):
    def click(self, position):
        # pyautogui.click(position[0], position[1])
        print("click", position)

autoplayer = AutoPlayer(parser=MyParser(), clicker=MyClicker())
autoplayer.step()
```

## 下一步建议

- 将 `VisionParser` 接到 OpenCV/YOLO 模型输出。
- 使用窗口句柄抓图（Win32/ADB/scrcpy）。
- 在 `GreedyAgent` 外挂 beam-search/MCTS 做 lookahead。
