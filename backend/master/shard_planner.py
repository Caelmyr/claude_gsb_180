"""Input sharding and task-granularity planning.

The planner turns a submitted job into concrete input shards and task objects.
It owns two difficulty points:

* **split strategy** — the user picks *how* the input is cut at submit time:
  ``count`` balances the number of records per shard (rows), ``size`` balances
  the byte volume per shard (data volume).  The chosen strategy is persisted
  on the job and every shard document, so the shards page always reflects
  exactly what the map tasks later consume — never a recomputed guess.
* **task-granularity / load-balancing** — the number of map tasks is clamped
  against the input size so a job never spawns a thousand empty tasks, and no
  strategy ever emits an empty shard while records remain.

Edge cases are handled in one place (the splitters) so the stored shards and
the shards the job actually processes can never drift apart:

* rows not divisible by the shard count — the remainder is dealt out to the
  first shards, so shard sizes differ by at most one record;
* a tiny or empty trailing shard — shard count is clamped to the record count
  and the size splitter keeps at least one record per remaining slot;
* strategy/data mismatch — an unknown strategy is rejected at submit time,
  and ``size`` on records with no measurable bytes falls back to ``count``
  (the fallback is recorded on the job and in the log).
"""

from __future__ import annotations

from typing import Any

from backend.common import constants as C
from backend.common.ids import shard_id
from backend.common.jsonutil import dumps_line, now_ms
from backend.common.models import Job, Task, new_task
from backend.common.storage import Storage
from backend.tasks.samples import generate_input_records


def record_size(record: Any) -> int:
    """Byte size of a single input record (UTF-8 encoded).

    Text records measure their own length; structured records (e.g. kv dicts)
    measure their JSON encoding.  Unmeasurable records report 0.
    """
    if record is None:
        return 0
    if isinstance(record, str):
        return len(record.encode("utf-8"))
    try:
        return len(dumps_line(record).encode("utf-8"))
    except (TypeError, ValueError):
        return 0


def split_evenly(items: list[Any], n: int) -> list[list[Any]]:
    """Split ``items`` into ``n`` chunks whose sizes differ by at most one."""
    if not items:
        return [[] for _ in range(max(1, n))]
    n = max(1, min(n, len(items)))
    base, rem = divmod(len(items), n)
    chunks: list[list[Any]] = []
    idx = 0
    for i in range(n):
        size = base + (1 if i < rem else 0)
        chunks.append(items[idx:idx + size])
        idx += size
    return chunks


def split_by_size(items: list[Any], sizes: list[int], n: int) -> list[list[Any]]:
    """Split ``items`` into at most ``n`` contiguous shards of balanced bytes.

    Records are walked in order and packed into a shard until its byte size
    reaches the fair share (``total / n``).  Guarantees:

    * shards stay contiguous and in order (concatenation reproduces input);
    * no empty shard while items remain — a shard is only closed early when
      enough records remain to fill the remaining slots;
    * fewer than ``n`` shards when a single record outweighs the fair share
      (a record is never cut in half);
    * degenerate inputs (no items, no measurable bytes) fall back to the
      count splitter so the two strategies degrade identically.
    """
    if not items:
        return split_evenly(items, n)
    n = max(1, min(n, len(items)))
    total = sum(sizes)
    if total <= 0:
        return split_evenly(items, n)
    target = total / n
    shards: list[list[Any]] = []
    buf: list[Any] = []
    buf_size = 0
    for i, item in enumerate(items):
        slots_left = n - len(shards)
        if (buf and slots_left > 1
                and buf_size + sizes[i] > target
                and len(items) - i >= slots_left - 1):
            shards.append(buf)
            buf, buf_size = [], 0
        buf.append(item)
        buf_size += sizes[i]
    if buf:
        shards.append(buf)
    return shards


class ShardPlanner:
    def __init__(self, storage: Storage, config) -> None:
        self.storage = storage
        self.config = config

    def _seed_for(self, job: Job) -> int:
        # A job-stable seed: the same job definition always yields the same data
        # (useful for reproducible demos), yet different jobs differ.
        return (self.config.seed + sum(ord(c) for c in job.job_id)) % (2 ** 31 - 1)

    def _split(self, job: Job, records: list[Any], sizes: list[int],
               num_map: int) -> tuple[str, list[list[Any]], str]:
        """Apply the job's split strategy; returns (strategy_used, chunks, note)."""
        strategy = job.split_strategy
        if strategy == C.SPLIT_BY_SIZE:
            if sum(sizes) <= 0:
                # Strategy/data mismatch: nothing measurable to balance, so the
                # size strategy degenerates to the count strategy.
                note = "split strategy 'size' not applicable: records have no " \
                       "measurable byte size; fell back to 'count'"
                return C.SPLIT_BY_COUNT, split_evenly(records, num_map), note
            return strategy, split_by_size(records, sizes, num_map), ""
        # Unknown strategies are rejected at submit time; anything else that
        # arrives here is treated as the count strategy.
        return C.SPLIT_BY_COUNT, split_evenly(records, num_map), ""

    def plan(self, job: Job) -> dict:
        """Generate input records, split them into shards, and build tasks."""
        kind = job.params.get("input_kind", "wordcount")
        rows = max(1, int(job.input_rows))
        records = generate_input_records(kind, rows, self._seed_for(job))
        sizes = [record_size(r) for r in records]

        # Granularity: never create more map tasks than there are input records.
        num_map = max(1, min(job.num_map_tasks, len(records)))
        strategy, chunks, note = self._split(job, records, sizes, num_map)
        job.num_map_tasks = len(chunks)
        job.split_strategy = strategy

        input_shards: list[str] = []
        pos = 0
        for i, chunk in enumerate(chunks):
            sid = shard_id("in", i)
            nbytes = sum(sizes[pos:pos + len(chunk)])
            pos += len(chunk)
            self.storage.write({
                "shard_id": sid,
                "job_id": job.job_id,
                "stage": C.STAGE_INPUT,
                "index": i,
                "split_strategy": strategy,
                "records": chunk,
                "count": len(chunk),
                "bytes": nbytes,
                "created_ms": now_ms(),
            }, "jobs", job.job_id, "shards", C.STAGE_INPUT, f"{sid}.json")
            input_shards.append(sid)

        map_tasks = [new_task(job, C.TASK_MAP, i) for i in range(len(chunks))]
        reduce_tasks = [new_task(job, C.TASK_REDUCE, p) for p in range(job.num_reduce_tasks)]

        return {
            "input_shards": input_shards,
            "map_tasks": map_tasks,
            "reduce_tasks": reduce_tasks,
            "total_records": len(records),
            "total_bytes": sum(sizes),
            "split_strategy": strategy,
            "note": note,
        }

    def load_input_shard(self, job_id: str, shard: str) -> list[Any]:
        """Return exactly the records persisted for ``shard`` at plan time.

        This is the single read path used to build map-task specs, so what a
        map task processes is always identical to what the shards page shows.
        """
        doc = self.storage.read("jobs", job_id, "shards", C.STAGE_INPUT, f"{shard}.json", default={})
        return list(doc.get("records", [])) if doc else []

    def input_shards(self, job: Job) -> list[dict]:
        out: list[dict] = []
        from backend.common.storage import list_files, read_json
        root = self.storage.path("jobs", job.job_id, "shards", C.STAGE_INPUT)
        for path in list_files(root, suffix=".json"):
            doc = read_json(path)
            if doc:
                out.append({
                    "shard_id": doc.get("shard_id"),
                    "index": doc.get("index"),
                    "count": doc.get("count", 0),
                    "bytes": doc.get("bytes"),
                    "split_strategy": doc.get("split_strategy"),
                    "stage": C.STAGE_INPUT,
                })
        out.sort(key=lambda d: d.get("index") if d.get("index") is not None else 0)
        return out
