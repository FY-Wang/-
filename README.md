# Triple-Match Stack Agent Skeleton

This repository contains a minimal Python agent scaffold for stack-based triple-match games such as《羊了个羊：星球》.

## Included

- `Tile` / `GameState` data model for:
  - layered tile set
  - visibility checks via cover graph
  - shelf capacity and auto-clear on 3 matches
- `GreedyAgent` baseline policy:
  - prioritize immediate triples
  - then pair completion
  - then visibility unlock potential
- `run_episode` loop to simulate full runs.

## Quick Start

```python
from agent import Tile, GameState, GreedyAgent, run_episode

tiles = {
    1: Tile(1, "carrot", 2, (0, 0)),
    2: Tile(2, "carrot", 1, (0, 0)),
    3: Tile(3, "carrot", 0, (0, 0)),
}

# key covers value(s): upper -> set(lower)
covers = {
    1: {2},
    2: {3},
    3: set(),
}

state = GameState(tiles=tiles, covers=covers, shelf_capacity=7)
agent = GreedyAgent()
print(run_episode(state, agent))
```

## Next Extensions

- Replace greedy scoring with beam search / MCTS.
- Add tool actions (shuffle, undo, revive).
- Plug in CV-based tile extraction from screenshots.
