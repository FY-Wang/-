from agent import (
    AutoPlayer,
    BeamSearchAgent,
    Detection,
    GameState,
    GeometricCoverInferer,
    GreedyAgent,
    MCTSAgent,
    OpenCVYOLOVisionParser,
    ParsedFrame,
    ParsedTile,
    Tile,
    WindowInfo,
    run_episode,
    select_window_by_title,
)


class DummyParser:
    def __init__(self, frame: ParsedFrame):
        self.frame = frame

    def capture_and_parse(self) -> ParsedFrame:
        return self.frame


class DummyClicker:
    def __init__(self):
        self.clicked = []

    def click(self, position):
        self.clicked.append(position)


class DummyFrameSource:
    def __init__(self, frame="dummy"):
        self.frame = frame

    def capture(self):
        return self.frame


class DummyDetector:
    def __init__(self, detections):
        self._detections = detections

    def detect(self, frame):
        return self._detections


def test_visibility_and_pick():
    tiles = {1: Tile(1, "a", 1, (0, 0)), 2: Tile(2, "a", 0, (0, 0))}
    state = GameState(tiles=tiles, covers={1: {2}, 2: set()})
    assert state.is_visible(1)
    assert not state.is_visible(2)
    state.pick(1)
    assert state.is_visible(2)


def test_auto_match_removes_triplet():
    tiles = {
        1: Tile(1, "a", 0, (0, 0)),
        2: Tile(2, "a", 0, (1, 0)),
        3: Tile(3, "a", 0, (2, 0)),
    }
    state = GameState(tiles=tiles, covers={1: set(), 2: set(), 3: set()})
    state.pick(1)
    state.pick(2)
    state.pick(3)
    assert len(state.shelf) == 0


def test_episode_win():
    tiles = {
        1: Tile(1, "a", 0, (0, 0)),
        2: Tile(2, "a", 0, (1, 0)),
        3: Tile(3, "a", 0, (2, 0)),
    }
    state = GameState(tiles=tiles, covers={1: set(), 2: set(), 3: set()})
    assert run_episode(state, GreedyAgent()) == "win"


def test_autoplayer_clicks_strategy_tile_position():
    frame = ParsedFrame(
        tiles=[
            ParsedTile(id=10, type="carrot", layer=1, position=(100, 200)),
            ParsedTile(id=20, type="carrot", layer=0, position=(110, 210)),
        ],
        covers={10: {20}, 20: set()},
    )
    autoplay = AutoPlayer(parser=DummyParser(frame), clicker=DummyClicker())
    clicker = autoplay.clicker

    chosen = autoplay.step()
    assert chosen == 10
    assert clicker.clicked == [(100, 200)]


def test_beam_and_mcts_choose_valid_visible_tile():
    tiles = {
        1: Tile(1, "a", 0, (0, 0)),
        2: Tile(2, "b", 0, (1, 0)),
        3: Tile(3, "a", 0, (2, 0)),
    }
    state = GameState(tiles=tiles, covers={1: set(), 2: set(), 3: set()})

    beam_choice = BeamSearchAgent(depth=2, beam_width=2).choose(state)
    mcts_choice = MCTSAgent(playouts_per_action=4, max_rollout_steps=8).choose(state)

    assert beam_choice in {1, 2, 3}
    assert mcts_choice in {1, 2, 3}


def test_yolo_parser_pipeline_with_dummy_detector():
    detections = [
        Detection("carrot", (0, 0, 50, 50), 0.99, layer=1),
        Detection("carrot", (10, 10, 60, 60), 0.99, layer=0),
    ]
    parser = OpenCVYOLOVisionParser(
        frame_source=DummyFrameSource(),
        detector=DummyDetector(detections),
        cover_inferer=GeometricCoverInferer(overlap_threshold=0.1),
    )

    frame = parser.capture_and_parse()
    assert len(frame.tiles) == 2
    assert 1 in frame.covers.get(0, set())


def test_select_window_by_title_matches_keyword():
    windows = [
        WindowInfo(hwnd=101, title="记事本"),
        WindowInfo(hwnd=202, title="羊了个羊 - 微信小游戏"),
    ]
    matched = select_window_by_title(windows, ["羊了个羊"])
    assert matched is not None
    assert matched.hwnd == 202


def test_select_window_by_title_returns_none_when_missing():
    windows = [WindowInfo(hwnd=101, title="记事本")]
    matched = select_window_by_title(windows, ["羊了个羊"])
    assert matched is None
