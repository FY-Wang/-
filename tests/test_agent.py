from agent import (
    AutoPlayer,
    ClickController,
    GameState,
    GreedyAgent,
    ParsedFrame,
    ParsedTile,
    Tile,
    VisionParser,
    run_episode,
)


class DummyParser(VisionParser):
    def __init__(self, frame: ParsedFrame):
        self.frame = frame

    def capture_and_parse(self) -> ParsedFrame:
        return self.frame


class DummyClicker(ClickController):
    def __init__(self):
        self.clicked = []

    def click(self, position):
        self.clicked.append(position)


def test_visibility_and_pick():
    tiles = {
        1: Tile(1, "a", 1, (0, 0)),
        2: Tile(2, "a", 0, (0, 0)),
    }
    covers = {1: {2}, 2: set()}
    state = GameState(tiles=tiles, covers=covers)

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
    covers = {1: set(), 2: set(), 3: set()}
    state = GameState(tiles=tiles, covers=covers)

    state.pick(1)
    state.pick(2)
    assert len(state.shelf) == 2
    state.pick(3)
    assert len(state.shelf) == 0


def test_episode_win():
    tiles = {
        1: Tile(1, "a", 0, (0, 0)),
        2: Tile(2, "a", 0, (1, 0)),
        3: Tile(3, "a", 0, (2, 0)),
    }
    covers = {1: set(), 2: set(), 3: set()}
    state = GameState(tiles=tiles, covers=covers)
    outcome = run_episode(state, GreedyAgent())
    assert outcome == "win"


def test_autoplayer_clicks_strategy_tile_position():
    frame = ParsedFrame(
        tiles=[
            ParsedTile(id=10, type="carrot", layer=1, position=(100, 200)),
            ParsedTile(id=20, type="carrot", layer=0, position=(110, 210)),
        ],
        covers={10: {20}, 20: set()},
    )
    parser = DummyParser(frame)
    clicker = DummyClicker()
    autoplay = AutoPlayer(parser=parser, clicker=clicker)

    chosen = autoplay.step()
    assert chosen == 10
    assert clicker.clicked == [(100, 200)]


def test_state_from_frame_drops_unclickable_tiles():
    frame = ParsedFrame(
        tiles=[
            ParsedTile(id=1, type="x", layer=0, position=(1, 1), clickable=True),
            ParsedTile(id=2, type="x", layer=0, position=(2, 2), clickable=False),
        ],
        covers={1: {2}, 2: set()},
    )
    state = AutoPlayer._state_from_frame(frame)
    assert set(state.tiles.keys()) == {1}
    assert state.covers[1] == set()
