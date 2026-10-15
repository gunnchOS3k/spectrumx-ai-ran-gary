"""Offline sync engine with LWW and vector-clock conflict policies.

Replaces placeholder last-write-wins stubs with deterministic merge rules.
Does not claim cloud multi-device account sync or CRDT completeness.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Callable
import time
import uuid


class ConflictPolicy(str, Enum):
    LWW = "last_write_wins"
    VECTOR_CLOCK = "vector_clock"


class SyncStatus(str, Enum):
    PENDING = "pending"
    SYNCED = "synced"
    CONFLICT = "conflict"
    RESOLVED = "resolved"


@dataclass(order=True)
class VectorClock:
    """Lamport-style vector clock keyed by replica id."""

    clocks: dict[str, int] = field(default_factory=dict)

    def tick(self, replica_id: str) -> "VectorClock":
        next_clocks = dict(self.clocks)
        next_clocks[replica_id] = next_clocks.get(replica_id, 0) + 1
        return VectorClock(clocks=next_clocks)

    def merge(self, other: "VectorClock") -> "VectorClock":
        keys = set(self.clocks) | set(other.clocks)
        return VectorClock(
            clocks={k: max(self.clocks.get(k, 0), other.clocks.get(k, 0)) for k in keys}
        )

    def happens_before(self, other: "VectorClock") -> bool:
        keys = set(self.clocks) | set(other.clocks)
        strictly_less = False
        for k in keys:
            a = self.clocks.get(k, 0)
            b = other.clocks.get(k, 0)
            if a > b:
                return False
            if a < b:
                strictly_less = True
        return strictly_less

    def concurrent_with(self, other: "VectorClock") -> bool:
        return (
            not self.happens_before(other)
            and not other.happens_before(self)
            and self.clocks != other.clocks
        )

    def to_dict(self) -> dict[str, int]:
        return dict(self.clocks)

    @classmethod
    def from_dict(cls, data: dict[str, int] | None) -> "VectorClock":
        return cls(clocks=dict(data or {}))


@dataclass
class SyncRecord:
    key: str
    value: Any
    replica_id: str
    wall_time_ms: int
    vector: VectorClock = field(default_factory=VectorClock)
    version: int = 1
    tombstone: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "value": deepcopy(self.value),
            "replica_id": self.replica_id,
            "wall_time_ms": self.wall_time_ms,
            "vector": self.vector.to_dict(),
            "version": self.version,
            "tombstone": self.tombstone,
        }


@dataclass
class ConflictInfo:
    key: str
    local: dict[str, Any]
    remote: dict[str, Any]
    policy: str
    winner: str | None
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _default_now_ms() -> int:
    return int(time.time() * 1000)


@dataclass
class OfflineSyncEngine:
    replica_id: str = field(default_factory=lambda: f"replica-{uuid.uuid4().hex[:8]}")
    policy: ConflictPolicy = ConflictPolicy.VECTOR_CLOCK
    store: dict[str, SyncRecord] = field(default_factory=dict)
    queue: list[SyncRecord] = field(default_factory=list)
    conflicts: list[ConflictInfo] = field(default_factory=list)
    now_ms: Callable[[], int] = field(default=_default_now_ms, repr=False)

    def put(self, key: str, value: Any) -> SyncRecord:
        existing = self.store.get(key)
        base = existing.vector if existing else VectorClock()
        vector = base.tick(self.replica_id)
        version = (existing.version + 1) if existing else 1
        record = SyncRecord(
            key=key,
            value=deepcopy(value),
            replica_id=self.replica_id,
            wall_time_ms=self.now_ms(),
            vector=vector,
            version=version,
        )
        self.store[key] = record
        self.queue.append(record)
        return record

    def delete(self, key: str) -> SyncRecord:
        existing = self.store.get(key)
        base = existing.vector if existing else VectorClock()
        record = SyncRecord(
            key=key,
            value=None,
            replica_id=self.replica_id,
            wall_time_ms=self.now_ms(),
            vector=base.tick(self.replica_id),
            version=(existing.version + 1) if existing else 1,
            tombstone=True,
        )
        self.store[key] = record
        self.queue.append(record)
        return record

    def pending(self) -> list[dict[str, Any]]:
        return [r.to_dict() for r in self.queue]

    def clear_queue(self) -> None:
        self.queue.clear()

    def _resolve_lww(self, local: SyncRecord, remote: SyncRecord) -> tuple[SyncRecord, ConflictInfo]:
        if remote.wall_time_ms > local.wall_time_ms:
            winner, reason, chosen = "remote", "remote_newer_wall_time", remote
        elif remote.wall_time_ms < local.wall_time_ms:
            winner, reason, chosen = "local", "local_newer_wall_time", local
        elif remote.replica_id > local.replica_id:
            winner, reason, chosen = "remote", "tie_break_replica_id", remote
        else:
            winner, reason, chosen = "local", "tie_break_replica_id", local
        info = ConflictInfo(
            key=local.key,
            local=local.to_dict(),
            remote=remote.to_dict(),
            policy=ConflictPolicy.LWW.value,
            winner=winner,
            reason=reason,
        )
        return chosen, info

    def _resolve_vector(
        self, local: SyncRecord, remote: SyncRecord
    ) -> tuple[SyncRecord, ConflictInfo | None]:
        if remote.vector.happens_before(local.vector) or remote.vector.clocks == local.vector.clocks:
            return local, None
        if local.vector.happens_before(remote.vector):
            return remote, None
        chosen, info = self._resolve_lww(local, remote)
        info.policy = ConflictPolicy.VECTOR_CLOCK.value
        info.reason = f"concurrent_then_{info.reason}"
        chosen = SyncRecord(
            key=chosen.key,
            value=deepcopy(chosen.value),
            replica_id=chosen.replica_id,
            wall_time_ms=chosen.wall_time_ms,
            vector=local.vector.merge(remote.vector).tick(self.replica_id),
            version=max(local.version, remote.version) + 1,
            tombstone=chosen.tombstone,
        )
        info.winner = "merged_lww_fallback"
        return chosen, info

    def apply_remote(self, remote_dict: dict[str, Any]) -> dict[str, Any]:
        remote = SyncRecord(
            key=remote_dict["key"],
            value=deepcopy(remote_dict.get("value")),
            replica_id=remote_dict["replica_id"],
            wall_time_ms=int(remote_dict["wall_time_ms"]),
            vector=VectorClock.from_dict(remote_dict.get("vector")),
            version=int(remote_dict.get("version", 1)),
            tombstone=bool(remote_dict.get("tombstone", False)),
        )
        local = self.store.get(remote.key)
        if local is None:
            self.store[remote.key] = remote
            return {
                "status": SyncStatus.SYNCED.value,
                "key": remote.key,
                "applied": "remote",
                "mock": False,
            }

        if self.policy == ConflictPolicy.LWW:
            chosen, info = self._resolve_lww(local, remote)
            self.store[remote.key] = chosen
            self.conflicts.append(info)
            return {
                "status": SyncStatus.RESOLVED.value,
                "key": remote.key,
                "conflict": info.to_dict(),
                "mock": False,
            }

        chosen, info = self._resolve_vector(local, remote)
        self.store[remote.key] = chosen
        if info is None:
            return {
                "status": SyncStatus.SYNCED.value,
                "key": remote.key,
                "applied": "causal",
                "mock": False,
            }
        self.conflicts.append(info)
        return {
            "status": SyncStatus.CONFLICT.value,
            "key": remote.key,
            "conflict": info.to_dict(),
            "resolved_value": chosen.value,
            "mock": False,
        }

    def sync_from_peer(self, peer_records: list[dict[str, Any]]) -> dict[str, Any]:
        results = [self.apply_remote(r) for r in peer_records]
        self.clear_queue()
        return {
            "replica_id": self.replica_id,
            "policy": self.policy.value,
            "results": results,
            "store_size": len(self.store),
            "conflicts": [c.to_dict() for c in self.conflicts],
            "mock": False,
        }

    def get(self, key: str) -> Any:
        rec = self.store.get(key)
        if rec is None or rec.tombstone:
            return None
        return deepcopy(rec.value)

    def snapshot(self) -> dict[str, Any]:
        return {
            "replica_id": self.replica_id,
            "policy": self.policy.value,
            "records": {k: v.to_dict() for k, v in self.store.items()},
            "pending": self.pending(),
            "conflicts": [c.to_dict() for c in self.conflicts],
            "mock": False,
        }
