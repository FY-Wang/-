from __future__ import annotations

import argparse
import time

from agent import (
    AdbClicker,
    AdbSource,
    AutoPlayer,
    AutoWin32WindowSource,
    BeamSearchAgent,
    GreedyAgent,
    MCTSAgent,
    MSSScreenSource,
    OpenCVYOLODetector,
    OpenCVYOLOVisionParser,
    PyAutoGuiClicker,
    Win32WindowSource,
    load_parsed_frame,
)


def build_strategy(name: str):
    if name == "greedy":
        return GreedyAgent()
    if name == "beam":
        return BeamSearchAgent(depth=3, beam_width=4)
    if name == "mcts":
        return MCTSAgent(playouts_per_action=12)
    raise ValueError(f"Unknown strategy: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="一键运行：羊了个羊视觉自动点击 agent")
    parser.add_argument("--strategy", choices=["greedy", "beam", "mcts"], default="beam")
    parser.add_argument("--mode", choices=["demo", "live"], default="demo")
    parser.add_argument("--frame-json", default="sample_frame.json", help="demo 模式使用的 ParsedFrame json")
    parser.add_argument("--source", choices=["screen", "window", "window-auto", "adb"], default="window-auto")
    parser.add_argument("--window-hwnd", type=int, default=0)
    parser.add_argument("--window-keywords", default="羊了个羊", help="窗口自动识别关键词，多个用逗号分隔")
    parser.add_argument("--weights", default="yolo_tiles.pt")
    parser.add_argument("--adb-serial", default=None)
    parser.add_argument("--interval", type=float, default=0.2)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()

    strategy = build_strategy(args.strategy)

    if args.mode == "demo":
        frame = load_parsed_frame(args.frame_json)

        class DemoParser:
            def capture_and_parse(self):
                return frame

        class PrintClicker:
            def click(self, position):
                print(f"click -> {position}")

        autoplay = AutoPlayer(DemoParser(), PrintClicker(), strategy)
        removed = set()
        for _ in range(args.steps):
            choice = autoplay.step(removed=removed)
            if choice is None:
                print("no action")
                break
            print(f"chosen tile id: {choice}")
            removed.add(choice)
            time.sleep(args.interval)
        return

    if args.source == "screen":
        source = MSSScreenSource()
        clicker = PyAutoGuiClicker()
    elif args.source == "window":
        if not args.window_hwnd:
            raise ValueError("--window-hwnd is required when --source window")
        source = Win32WindowSource(args.window_hwnd)
        clicker = PyAutoGuiClicker()
    elif args.source == "window-auto":
        keywords = [k.strip() for k in args.window_keywords.split(",") if k.strip()]
        source = AutoWin32WindowSource(keywords=keywords)
        clicker = PyAutoGuiClicker()
    else:
        source = AdbSource(serial=args.adb_serial)
        clicker = AdbClicker(serial=args.adb_serial)

    detector = OpenCVYOLODetector(args.weights)
    vision = OpenCVYOLOVisionParser(source, detector)
    autoplay = AutoPlayer(vision, clicker, strategy)

    for _ in range(args.steps):
        choice = autoplay.step()
        print(f"chosen tile id: {choice}")
        if choice is None:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
