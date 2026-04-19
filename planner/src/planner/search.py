from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from typing import Any, Callable, Iterable, Protocol


@dataclass(frozen=True)
class SearchNode:
    node_id: str
    state: dict[str, Any]
    score: float
    depth: int = 0
    action: str | None = None
    parent_id: str | None = None


@dataclass(frozen=True)
class SearchResult:
    best_node: SearchNode
    path: tuple[SearchNode, ...]
    visited: tuple[SearchNode, ...]
    expansions: int


ExpandFn = Callable[[SearchNode], Iterable[SearchNode]]
GoalFn = Callable[[SearchNode], bool]


class SearchPolicy(Protocol):
    def search(self, root: SearchNode, expand: ExpandFn, is_terminal: GoalFn) -> SearchResult:
        ...


@dataclass
class BestFirstSearchPolicy:
    max_expansions: int = 64
    beam_width: int = 8

    def search(self, root: SearchNode, expand: ExpandFn, is_terminal: GoalFn) -> SearchResult:
        frontier: list[tuple[float, int, SearchNode]] = []
        nodes = {root.node_id: root}
        parents = {root.node_id: None}
        visited: list[SearchNode] = []
        best = root
        order = 0

        heappush(frontier, (-root.score, order, root))
        order += 1
        expansions = 0

        while frontier and expansions < self.max_expansions:
            _, _, node = heappop(frontier)
            visited.append(node)
            if node.score > best.score:
                best = node
            if is_terminal(node):
                best = node
                break

            children = sorted(expand(node), key=lambda item: item.score, reverse=True)
            for child in children[: self.beam_width]:
                nodes[child.node_id] = child
                parents[child.node_id] = node.node_id
                heappush(frontier, (-child.score, order, child))
                order += 1
            expansions += 1

        return SearchResult(
            best_node=best,
            path=_reconstruct_path(best.node_id, nodes, parents),
            visited=tuple(visited),
            expansions=expansions,
        )


def _reconstruct_path(
    node_id: str,
    nodes: dict[str, SearchNode],
    parents: dict[str, str | None],
) -> tuple[SearchNode, ...]:
    path: list[SearchNode] = []
    current: str | None = node_id
    while current is not None:
        path.append(nodes[current])
        current = parents.get(current)
    return tuple(reversed(path))
