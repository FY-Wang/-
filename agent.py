from __future__ import annotations

import json
import random
import subprocess
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Set, Tuple

Position = Tuple[int, int]
BBox = Tuple[int, int, int, int]


@dataclass(frozen=True)
class Tile:
    id: int
    type: str
    layer: int
    position: Position


@dataclass
class GameState:
    tiles: Dict[int, Tile]
    covers: Dict[int, Set[int]]
    shelf_capacity: int = 7
    shelf: List[Tile] = field(default_factory=list)
    removed: Set[int] = field(default_factory=set)
    history: List[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        for tile_id in self.tiles:
            self.covers.setdefault(tile_id, set())

    def clone(self) -> "GameState":
        return GameState(
            tiles=self.tiles,
            covers=self.covers,
            shelf_capacity=self.shelf_capacity,
            shelf=list(self.shelf),
            removed=set(self.removed),
            history=list(self.history),
        )

    @property
    def remaining(self) -> int:
        return len(self.tiles) - len(self.removed)

    @property
    def won(self) -> bool:
        return self.remaining == 0

    @property
    def lost(self) -> bool:
        return len(self.shelf) > self.shelf_capacity

    def blockers_of(self, tile_id: int) -> Set[int]:
        blockers: Set[int] = set()
        for upper, lowers in self.covers.items():
            if tile_id in lowers and upper not in self.removed:
                blockers.add(upper)
        return blockers

    def is_visible(self, tile_id: int) -> bool:
        return tile_id not in self.removed and not self.blockers_of(tile_id)

    def visible_tiles(self) -> List[Tile]:
        return [self.tiles[tile_id] for tile_id in self.tiles if self.is_visible(tile_id)]

    def pick(self, tile_id: int) -> None:
        if not self.is_visible(tile_id):
            raise ValueError(f"Tile {tile_id} is not visible.")
        tile = self.tiles[tile_id]
        self.removed.add(tile_id)
        self.shelf.append(tile)
        self.history.append(tile_id)
        self._auto_match(tile.type)

    def _auto_match(self, tile_type: str) -> None:
        matches = [i for i, tile in enumerate(self.shelf) if tile.type == tile_type]
        if len(matches) < 3:
            return
        remove_idx = set(matches[:3])
        self.shelf = [tile for i, tile in enumerate(self.shelf) if i not in remove_idx]

    def shelf_counts(self) -> Counter:
        return Counter(tile.type for tile in self.shelf)


ScoreFn = Callable[[GameState, Tile], float]


class StrategyAgent(Protocol):
    def choose(self, state: GameState) -> Optional[int]:
        ...


class GreedyAgent:
    def __init__(self, extra_score_fn: Optional[ScoreFn] = None):
        self.extra_score_fn = extra_score_fn

    def tile_score(self, state: GameState, tile: Tile) -> float:
        shelf_counts = state.shelf_counts()
        count = shelf_counts[tile.type]
        immediate_match_bonus = 100.0 if count == 2 else 0.0
        pair_bonus = 10.0 if count == 1 else 0.0
        unlock_bonus = self._unlocking_score(state, tile.id)
        risk_penalty = 5.0 if len(state.shelf) >= state.shelf_capacity - 1 and count == 0 else 0.0
        extra = self.extra_score_fn(state, tile) if self.extra_score_fn else 0.0
        return immediate_match_bonus + pair_bonus + unlock_bonus + extra - risk_penalty

    def choose(self, state: GameState) -> Optional[int]:
        visible = state.visible_tiles()
        if not visible:
            return None
        return max(visible, key=lambda t: self.tile_score(state, t)).id

    @staticmethod
    def _unlocking_score(state: GameState, chosen_tile_id: int) -> float:
        unlocked = 0
        for lower_id in state.covers.get(chosen_tile_id, set()):
            if lower_id in state.removed:
                continue
            if state.blockers_of(lower_id) == {chosen_tile_id}:
                unlocked += 1
        return float(unlocked)


class BeamSearchAgent:
    def __init__(self, base_agent: Optional[GreedyAgent] = None, depth: int = 3, beam_width: int = 4):
        self.base_agent = base_agent or GreedyAgent()
        self.depth = depth
        self.beam_width = beam_width

    def choose(self, state: GameState) -> Optional[int]:
        visible = state.visible_tiles()
        if not visible:
            return None

        best_first_action: Optional[int] = None
        best_value = -float("inf")
        for first in visible:
            rollout = state.clone()
            rollout.pick(first.id)
            value = self._search(rollout, self.depth - 1)
            if value > best_value:
                best_value = value
                best_first_action = first.id
        return best_first_action

    def _search(self, state: GameState, depth_left: int) -> float:
        if state.won:
            return 10000.0
        if state.lost:
            return -10000.0
        if depth_left <= 0:
            return self._state_value(state)

        candidates = state.visible_tiles()
        if not candidates:
            return self._state_value(state)

        ranked = sorted(candidates, key=lambda t: self.base_agent.tile_score(state, t), reverse=True)[: self.beam_width]
        best = -float("inf")
        for tile in ranked:
            nxt = state.clone()
            nxt.pick(tile.id)
            best = max(best, self._search(nxt, depth_left - 1))
        return best

    @staticmethod
    def _state_value(state: GameState) -> float:
        return -3.0 * len(state.shelf) - state.remaining


class MCTSAgent:
    def __init__(self, playouts_per_action: int = 20, max_rollout_steps: int = 64):
        self.playouts_per_action = playouts_per_action
        self.max_rollout_steps = max_rollout_steps
        self.greedy = GreedyAgent()

    def choose(self, state: GameState) -> Optional[int]:
        visible = state.visible_tiles()
        if not visible:
            return None

        best_action: Optional[int] = None
        best_score = -float("inf")
        for action in visible:
            total = 0.0
            for _ in range(self.playouts_per_action):
                sim = state.clone()
                sim.pick(action.id)
                total += self._rollout(sim)
            avg = total / self.playouts_per_action
            if avg > best_score:
                best_score = avg
                best_action = action.id
        return best_action

    def _rollout(self, state: GameState) -> float:
        for _ in range(self.max_rollout_steps):
            if state.won:
                return 1000.0
            if state.lost:
                return -1000.0
            visible = state.visible_tiles()
            if not visible:
                break
            if random.random() < 0.7:
                action = self.greedy.choose(state)
            else:
                action = random.choice(visible).id
            if action is None:
                break
            state.pick(action)
        return -len(state.shelf) - state.remaining


@dataclass(frozen=True)
class ParsedTile:
    id: int
    type: str
    layer: int
    position: Position
    bbox: Optional[BBox] = None
    clickable: bool = True


@dataclass
class ParsedFrame:
    tiles: List[ParsedTile]
    covers: Dict[int, Set[int]]


class VisionParser(Protocol):
    def capture_and_parse(self) -> ParsedFrame:
        ...


class ClickController(Protocol):
    def click(self, position: Position) -> None:
        ...


class FrameSource(Protocol):
    def capture(self) -> Any:
        ...


@dataclass(frozen=True)
class Detection:
    label: str
    bbox: BBox
    confidence: float
    layer: int = 0


class Detector(Protocol):
    def detect(self, frame: Any) -> List[Detection]:
        ...


class CoverInferer(Protocol):
    def infer(self, detections: Sequence[Detection]) -> Dict[int, Set[int]]:
        ...


class GeometricCoverInferer:
    def __init__(self, overlap_threshold: float = 0.15):
        self.overlap_threshold = overlap_threshold

    def infer(self, detections: Sequence[Detection]) -> Dict[int, Set[int]]:
        covers: Dict[int, Set[int]] = {i: set() for i in range(len(detections))}
        for i, a in enumerate(detections):
            for j, b in enumerate(detections):
                if i == j:
                    continue
                if a.layer <= b.layer:
                    continue
                if self._overlap_ratio(a.bbox, b.bbox) >= self.overlap_threshold:
                    covers[i].add(j)
        return covers

    @staticmethod
    def _overlap_ratio(a: BBox, b: BBox) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        if inter == 0:
            return 0.0
        area_b = max(1, (bx2 - bx1) * (by2 - by1))
        return inter / area_b


class OpenCVYOLODetector:
    def __init__(self, model_path: str):
        from ultralytics import YOLO  # type: ignore

        self._model = YOLO(model_path)

    def detect(self, frame: Any) -> List[Detection]:
        results = self._model.predict(frame, verbose=False)
        detections: List[Detection] = []
        for result in results:
            names = result.names
            for box in result.boxes:
                cls = int(box.cls.item())
                conf = float(box.conf.item())
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                detections.append(Detection(label=str(names[cls]), bbox=(x1, y1, x2, y2), confidence=conf, layer=0))
        return detections


@dataclass(frozen=True)
class WindowInfo:
    hwnd: int
    title: str


def select_window_by_title(windows: Sequence[WindowInfo], keywords: Sequence[str]) -> Optional[WindowInfo]:
    normalized_keywords = [k.lower() for k in keywords if k]
    if not normalized_keywords:
        return None
    for info in windows:
        title = info.title.lower()
        if any(k in title for k in normalized_keywords):
            return info
    return None


def find_window_by_title_keywords(keywords: Sequence[str]) -> Optional[WindowInfo]:
    import win32gui  # type: ignore

    windows: List[WindowInfo] = []

    def callback(hwnd: int, _ctx: Any) -> None:
        if not win32gui.IsWindowVisible(hwnd):
            return
        title = win32gui.GetWindowText(hwnd)
        if not title:
            return
        windows.append(WindowInfo(hwnd=hwnd, title=title))

    win32gui.EnumWindows(callback, None)
    return select_window_by_title(windows, keywords)


class MSSScreenSource:
    def __init__(self, monitor: int = 1, region: Optional[Dict[str, int]] = None):
        self.monitor = monitor
        self.region = region

    def capture(self) -> Any:
        import mss  # type: ignore
        import numpy as np  # type: ignore

        with mss.mss() as sct:
            target = self.region or sct.monitors[self.monitor]
            shot = sct.grab(target)
            return np.array(shot)[:, :, :3]


class Win32WindowSource:
    def __init__(self, hwnd: int):
        self.hwnd = hwnd

    def capture(self) -> Any:
        import mss  # type: ignore
        import numpy as np  # type: ignore
        import win32gui  # type: ignore

        left, top, right, bottom = win32gui.GetWindowRect(self.hwnd)
        region = {"left": left, "top": top, "width": right - left, "height": bottom - top}
        with mss.mss() as sct:
            shot = sct.grab(region)
            return np.array(shot)[:, :, :3]


class AutoWin32WindowSource:
    """Automatically locate a visible window by title keywords and capture it."""

    def __init__(self, keywords: Optional[Sequence[str]] = None):
        self.keywords = list(keywords or ["羊了个羊"])

    def capture(self) -> Any:
        matched = find_window_by_title_keywords(self.keywords)
        if matched is None:
            raise RuntimeError(f"No visible window matched keywords: {self.keywords}")
        return Win32WindowSource(matched.hwnd).capture()


class AdbSource:
    def __init__(self, adb_path: str = "adb", serial: Optional[str] = None):
        self.adb_path = adb_path
        self.serial = serial

    def _adb_prefix(self) -> List[str]:
        cmd = [self.adb_path]
        if self.serial:
            cmd += ["-s", self.serial]
        return cmd

    def capture(self) -> Any:
        import cv2  # type: ignore
        import numpy as np  # type: ignore

        cmd = self._adb_prefix() + ["exec-out", "screencap", "-p"]
        data = subprocess.check_output(cmd)
        array = np.frombuffer(data, dtype=np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("Failed to decode adb screenshot")
        return image


class OpenCVYOLOVisionParser:
    def __init__(self, frame_source: FrameSource, detector: Detector, cover_inferer: Optional[CoverInferer] = None, conf_th: float = 0.25):
        self.frame_source = frame_source
        self.detector = detector
        self.cover_inferer = cover_inferer or GeometricCoverInferer()
        self.conf_th = conf_th

    def capture_and_parse(self) -> ParsedFrame:
        frame = self.frame_source.capture()
        detections = [d for d in self.detector.detect(frame) if d.confidence >= self.conf_th]
        covers_idx = self.cover_inferer.infer(detections)

        parsed_tiles: List[ParsedTile] = []
        for i, d in enumerate(detections):
            x1, y1, x2, y2 = d.bbox
            parsed_tiles.append(
                ParsedTile(
                    id=i,
                    type=d.label,
                    layer=d.layer,
                    position=((x1 + x2) // 2, (y1 + y2) // 2),
                    bbox=d.bbox,
                    clickable=True,
                )
            )

        covers: Dict[int, Set[int]] = {i: set(v) for i, v in covers_idx.items()}
        return ParsedFrame(tiles=parsed_tiles, covers=covers)


class PyAutoGuiClicker:
    def click(self, position: Position) -> None:
        import pyautogui  # type: ignore

        pyautogui.click(position[0], position[1])


class AdbClicker:
    def __init__(self, adb_path: str = "adb", serial: Optional[str] = None):
        self.adb_path = adb_path
        self.serial = serial

    def click(self, position: Position) -> None:
        x, y = position
        cmd = [self.adb_path]
        if self.serial:
            cmd += ["-s", self.serial]
        cmd += ["shell", "input", "tap", str(x), str(y)]
        subprocess.check_call(cmd)


class AutoPlayer:
    def __init__(self, parser: VisionParser, clicker: ClickController, agent: Optional[StrategyAgent] = None):
        self.parser = parser
        self.clicker = clicker
        self.agent = agent or GreedyAgent()

    def step(self, shelf: Optional[List[Tile]] = None, removed: Optional[Set[int]] = None) -> Optional[int]:
        frame = self.parser.capture_and_parse()
        state = self._state_from_frame(frame, shelf=shelf, removed=removed)
        choice = self.agent.choose(state)
        if choice is None:
            return None
        self.clicker.click(state.tiles[choice].position)
        return choice

    @staticmethod
    def _state_from_frame(frame: ParsedFrame, shelf: Optional[List[Tile]] = None, removed: Optional[Set[int]] = None) -> GameState:
        tiles = {
            tile.id: Tile(id=tile.id, type=tile.type, layer=tile.layer, position=tile.position)
            for tile in frame.tiles
            if tile.clickable
        }
        covers: Dict[int, Set[int]] = {tile_id: set() for tile_id in tiles}
        for upper, lowers in frame.covers.items():
            if upper not in tiles:
                continue
            covers[upper] = {lower for lower in lowers if lower in tiles}

        return GameState(tiles=tiles, covers=covers, shelf=list(shelf or []), removed=set(removed or set()))


def run_episode(state: GameState, agent: StrategyAgent, max_steps: int = 10_000) -> str:
    for _ in range(max_steps):
        if state.won:
            return "win"
        if state.lost:
            return "loss"
        action = agent.choose(state)
        if action is None:
            return "stuck"
        state.pick(action)
    return "stuck"


def load_parsed_frame(path: str) -> ParsedFrame:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    tiles = [
        ParsedTile(
            id=t["id"],
            type=t["type"],
            layer=t.get("layer", 0),
            position=tuple(t["position"]),
            bbox=tuple(t["bbox"]) if t.get("bbox") else None,
            clickable=t.get("clickable", True),
        )
        for t in payload["tiles"]
    ]
    covers = {int(k): set(v) for k, v in payload.get("covers", {}).items()}
    return ParsedFrame(tiles=tiles, covers=covers)
