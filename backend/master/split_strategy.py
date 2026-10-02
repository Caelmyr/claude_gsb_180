"""Pluggable input split strategies.

A split strategy turns the generated input records into *contiguous* shards.
Every strategy lives in :data:`STRATEGIES` so the set is configurable/extensible
from one place, and **all strategies share the same boundary rules**:

* an empty input always yields exactly one (empty) shard — a job always has at
  least one map task, never an empty task list;
* the shard count is clamped to ``min(requested, len(records))`` so a request
  for 100 shards over 3 records produces 3 one-record shards, not 97 empty
  ones (the clamp is reported back as a warning);
* records are never split or reordered — each input record belongs to exactly
  one shard, and shard ``i`` feeds map task ``i``.

Two strategies ship by default:

* ``rows``  — equal *record count* per shard (counts differ by at most one);
* ``bytes`` — balanced *serialized byte size* per shard, for inputs whose
  records vary a lot in size (e.g. long lines vs. short ones).

The byte weight of a record is its compact UTF-8 JSON length
(:func:`record_size_bytes`); the shard page displays shard sizes with exactly
this same measure, so what is shown is what the map task processes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

from backend.common import constants as C
from backend.common.jsonutil import dumps_line

SPLIT_ROWS = C.SPLIT_STRATEGY_ROWS
SPLIT_BYTES = C.SPLIT_STRATEGY_BYTES
DEFAULT_STRATEGY = C.DEFAULT_SPLIT_STRATEGY


def record_size_bytes(record: Any) -> int:
    """Byte weight of one record: its compact JSON encoding in UTF-8.

    This single function defines "shard size" everywhere (byte-based splitting
    and the shard page display), so the two can never disagree.
    """
    return len(dumps_line(record).encode("utf-8"))


# ---------------------------------------------------------------------------
# Shared boundary handling
# ---------------------------------------------------------------------------
def clamp_shard_count(num_records: int, requested: int) -> tuple[int, list[str]]:
    """Apply the universal shard-count rules, returning ``(n, warnings)``."""
    warnings: list[str] = []
    try:
        requested = max(1, int(requested))
    except (TypeError, ValueError):
        requested = 1
    if num_records == 0:
        warnings.append("输入为空，仅产生 1 个空分片（empty input → a single empty shard）")
        return 1, warnings
    n = min(requested, num_records)
    if n < requested:
        warnings.append(
            f"请求 {requested} 个分片但仅有 {num_records} 条输入，"
            f"实际产生 {n} 个分片（clamped to one record per shard）"
        )
    return n, warnings


def split_by_rows(records: list[Any], requested: int) -> tuple[list[list[Any]], list[str]]:
    """Split into ``n`` chunks whose record counts differ by at most one.

    The remainder is spread over the first ``rem`` chunks (each gets one extra
    record), so a non-divisible row count never leaves a tiny last shard.
    """
    n, warnings = clamp_shard_count(len(records), requested)
    if not records:
        return [[]], warnings
    base, rem = divmod(len(records), n)
    chunks: list[list[Any]] = []
    idx = 0
    for i in range(n):
        size = base + (1 if i < rem else 0)
        chunks.append(records[idx:idx + size])
        idx += size
    return chunks, warnings


def split_by_bytes(records: list[Any], requested: int) -> tuple[list[list[Any]], list[str]]:
    """Split so every shard has roughly equal serialized byte size.

    Records are packed contiguously against an even byte target; a single
    oversized record always lands whole in one shard, which may make that
    shard larger than the target (reported as a warning, never silently
    reshuffled). If the greedy pass cannot open ``n`` shards (a few huge
    records dominate), the biggest shards are split in half repeatedly until
    the count is reached, guaranteeing non-empty shards.
    """
    n, warnings = clamp_shard_count(len(records), requested)
    if not records:
        return [[]], warnings

    sizes = [record_size_bytes(r) for r in records]
    total = sum(sizes)
    target = max(1, math.ceil(total / n))

    # Greedy contiguous packing: close a shard when the next record would
    # overflow the even target, but never close the last shard early.
    chunks: list[list[Any]] = [[]]
    chunk_sizes: list[int] = [0]
    for rec, size in zip(records, sizes):
        if chunk_sizes[-1] > 0 and chunk_sizes[-1] + size > target and len(chunks) < n:
            chunks.append([])
            chunk_sizes.append(0)
        chunks[-1].append(rec)
        chunk_sizes[-1] += size

    # Refill to exactly n non-empty shards by halving the largest shard.
    while len(chunks) < n:
        biggest = max(range(len(chunks)), key=lambda i: len(chunks[i]))
        if len(chunks[biggest]) < 2:
            break  # every shard already holds a single record
        mid = len(chunks[biggest]) // 2
        left, right = chunks[biggest][:mid], chunks[biggest][mid:]
        chunks[biggest:biggest + 1] = [left, right]
        chunk_sizes[biggest:biggest + 1] = [
            sum(record_size_bytes(r) for r in left),
            sum(record_size_bytes(r) for r in right),
        ]

    final_sizes = [sum(record_size_bytes(r) for r in c) for c in chunks]
    oversized = [s for s in final_sizes if s > target * 1.5]
    if oversized:
        warnings.append(
            f"存在超过均摊目标（{target} B）的大记录，{len(oversized)} 个分片会大于平均大小；"
            "记录不拆分，按数据量切分保持记录完整（oversized records kept whole）"
        )
    return chunks, warnings


@dataclass(frozen=True)
class SplitStrategy:
    name: str
    label: str
    description: str
    fn: Callable[[list[Any], int], tuple[list[list[Any]], list[str]]]
    size_basis: str = "records"  # "records" (count) or "bytes"


STRATEGIES: dict[str, SplitStrategy] = {
    SPLIT_ROWS: SplitStrategy(
        name=SPLIT_ROWS,
        label="按行数均分 Even by row count",
        description=(
            "每个分片的记录数尽量相等（相差不超过 1 条），适合每条记录大小相近的文本类输入。"
            " Record counts per shard differ by at most one; best for uniformly sized records."
        ),
        fn=split_by_rows,
        size_basis="records",
    ),
    SPLIT_BYTES: SplitStrategy(
        name=SPLIT_BYTES,
        label="按数据量均衡 Even by byte size",
        description=(
            "按记录序列化后的字节总量均衡分片，适合记录长短差异大的输入；记录不会被拆分。"
            " Balances serialized bytes per shard for skewed record sizes; records stay whole."
        ),
        fn=split_by_bytes,
        size_basis="bytes",
    ),
}


def is_strategy(name: str) -> bool:
    return name in STRATEGIES


def get_strategy(name: str) -> SplitStrategy:
    """Look up a strategy, falling back to the default for an unknown name."""
    return STRATEGIES.get(name or "", STRATEGIES[DEFAULT_STRATEGY])


def apply_strategy(name: str, records: list[Any], requested: int):
    """Split ``records`` with the named strategy. See :class:`SplitStrategy`."""
    strategy = get_strategy(name)
    chunks, warnings = strategy.fn(records, requested)
    return strategy, chunks, warnings


def list_strategies() -> list[dict]:
    """JSON-serialisable strategy metadata for the API / submit page."""
    return [
        {
            "name": s.name,
            "label": s.label,
            "description": s.description,
            "size_basis": s.size_basis,
            "default": s.name == DEFAULT_STRATEGY,
        }
        for s in STRATEGIES.values()
    ]
