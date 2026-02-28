from agent import GameState, GreedyAgent, Tile, run_episode


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
