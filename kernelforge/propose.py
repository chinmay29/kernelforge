"""Mutation proposal strategy."""

from __future__ import annotations

import random
from dataclasses import dataclass

from kernelforge.mutations import MUTATION_REGISTRY, MutationResult, apply_mutation


@dataclass(frozen=True)
class Proposal:
    mutations: list[MutationResult]
    source: str


class Proposer:
    def __init__(self, seed: int) -> None:
        self.rng = random.Random(seed)

    def propose(self, source: str, retrieved: list[dict], max_mutations: int = 2) -> Proposal:
        preferred: list[str] = []
        for row in retrieved:
            for name in row.get("mutations", []):
                if name in MUTATION_REGISTRY and name not in preferred:
                    preferred.append(name)
        if not preferred:
            preferred = list(MUTATION_REGISTRY.keys())

        n = self.rng.randint(1, max_mutations)
        chosen: list[str] = []
        pool = preferred[:]
        self.rng.shuffle(pool)
        for name in pool:
            if len(chosen) >= n:
                break
            chosen.append(name)

        if len(chosen) < n:
            extra = [x for x in MUTATION_REGISTRY if x not in chosen]
            self.rng.shuffle(extra)
            chosen.extend(extra[: n - len(chosen)])

        current = source
        results: list[MutationResult] = []
        for name in chosen:
            res = apply_mutation(name, current, self.rng)
            current = res.source
            results.append(res)

        return Proposal(mutations=results, source=current)

