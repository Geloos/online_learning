"""Concept pool: frozen ensembles of past concepts, FIFO eviction.

Stored ensembles never learn. Recall works on a copy; a recalled ensemble
that wins its episode overwrites its own entry (no duplicate)."""

from __future__ import annotations


class ConceptPool:
    def __init__(self, max_size: int):
        self.max_size = max_size
        self.entries: list[dict] = []  # oldest first
        self._next_id = 0

    def __len__(self) -> int:
        return len(self.entries)

    def add(self, ensemble, step: int) -> tuple[int | None, int | None]:
        """Store a frozen ensemble. Returns (new id, evicted id). With
        max_size 0 the pool is disabled and nothing is stored."""
        if self.max_size <= 0:
            return None, None
        evicted = None
        if len(self.entries) >= self.max_size:
            evicted = self.entries.pop(0)["id"]
        entry_id = self._next_id
        self._next_id += 1
        self.entries.append({"id": entry_id, "ensemble": ensemble,
                             "created_step": step, "updated_step": step})
        return entry_id, evicted

    def get(self, entry_id: int) -> dict | None:
        for e in self.entries:
            if e["id"] == entry_id:
                return e
        return None

    def replace(self, entry_id: int, ensemble, step: int) -> bool:
        """Overwrite an entry's ensemble in place (its FIFO position is
        kept). False if the entry was evicted meanwhile."""
        e = self.get(entry_id)
        if e is None:
            return False
        e["ensemble"] = ensemble
        e["updated_step"] = step
        return True

    def ids(self) -> list[int]:
        return [e["id"] for e in self.entries]
