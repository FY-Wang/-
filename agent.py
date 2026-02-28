from __future__ import annotations

from dataclasses import dataclass, field
from collections import Counter, defaultdict
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

Position = Tuple[int, int]


@dataclass(frozen=True)
class Tile:
    """A single stackable tile in the puzzle."""

    id: int
    type: str
    layer: int
    position: Position


@dataclass
class GameState:
    """Runtime state for a triple-match stack game."""

    tiles: Dict[int, Tile]
    covers: Dict[int, Set[int]]
    shelf_capacity: int = 7
    shelf: List[Tile] = field(default_factory=list)
    removed: Set[int] = field(default_factory=set)
    history: List[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        for tile_id in self.tiles:
            self.covers.setdefault(tile_id, set())

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
        matches = [index for index, tile in enumerate(self.shelf) if tile.type == tile_type]
        if len(matches) < 3:
            return
        # Remove exactly three of the type to mimic a 3-match clear.
        to_remove = set(matches[:3])
        self.shelf = [tile for index, tile in enumerate(self.shelf) if index not in to_remove]

    def shelf_counts(self) -> Counter:
        return Counter(tile.type for tile in self.shelf)


ScoreFn = Callable[[GameState, Tile], float]


class GreedyAgent:
    """
    Baseline policy:
    1. Finish immediate triples first.
    2. Favor tiles whose type already exists in shelf.
    3. Favor actions that unlock more future visible tiles.
    """

    def __init__(self, extra_score_fn: Optional[ScoreFn] = None):
        self.extra_score_fn = extra_score_fn

    def choose(self, state: GameState) -> Optional[int]:
        visible = state.visible_tiles()
        if not visible:
            return None

        shelf_counts = state.shelf_counts()

        def score(tile: Tile) -> float:
            count = shelf_counts[tile.type]
            immediate_match_bonus = 100.0 if count == 2 else 0.0
            pair_bonus = 10.0 if count == 1 else 0.0
            unlock_bonus = self._unlocking_score(state, tile.id)
            capacity_risk_penalty = 5.0 if len(state.shelf) >= state.shelf_capacity - 1 and count == 0 else 0.0
            extra = self.extra_score_fn(state, tile) if self.extra_score_fn else 0.0
            return immediate_match_bonus + pair_bonus + unlock_bonus + extra - capacity_risk_penalty

        chosen = max(visible, key=score)
        return chosen.id

    @staticmethod
    def _unlocking_score(state: GameState, chosen_tile_id: int) -> float:
        """Estimate how many lower tiles become newly visible if chosen tile is removed."""
        unlocked = 0
        for lower_id in state.covers.get(chosen_tile_id, set()):
            if lower_id in state.removed:
                continue
            blockers = state.blockers_of(lower_id)
            if blockers == {chosen_tile_id}:
                unlocked += 1
        return float(unlocked)


def run_episode(state: GameState, agent: GreedyAgent, max_steps: int = 10_000) -> str:
    """Run until win/loss/stuck. Returns one of: 'win', 'loss', 'stuck'."""
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
