"""Input sharding and task-granularity planning.

The planner turns a submitted job into concrete input shards and task objects.
It also owns the **task-granularity / load-balancing** difficulty point: the
number of map tasks is clamped against the input size so a job never spawns a
thousand empty tasks, and records are split with the job's chosen
:mod:`~backend.master.split_strategy`.

Consistency model
-----------------
The actual sharding is performed exactly once, at submission time.  The same
chunks are:

* persisted as shard documents (the bytes the workers read and process), and
* summarised in a ``split_plan.json`` manifest (count + byte size per shard),

so the shards page always shows precisely what the map tasks receive — the
strategy is locked in at submission and the displayed shard counts/sizes are
read back from the produced artifacts, never recomputed.
"""

from __future__ import annotations

from typing import Any

from backend.common import constants as C
from backend.common.ids import shard_id
from backend.common.jsonutil import now_ms
from backend.common.models import Job, Task, new_task
from backend.common.storage import Storage, list_files, read_json
from backend.master.split_strategy import (
    DEFAULT_STRATEGY,
    apply_strategy,
    get_strategy,
    record_size_bytes,
)
from backend.tasks.samples import generate_input_records


def split_evenly(items: list[Any], n: int) -> list[list[Any]]:
    """Split ``items`` into ``n`` chunks whose sizes differ by at most one.

    Boundary rules (kept for direct callers/tests and identical to the
    ``rows`` strategy): empty input gives one empty chunk; more requested
    parts than items clamps to one item per part.
    """
    if not items:
        return [[]]
    n = max(1, min(n, len(items)))
    base, rem = divmod(len(items), n)
    chunks: list[list[Any]] = []
    idx = 0
    for i in range(n):
        size = base + (1 if i < rem else 0)
        chunks.append(items[idx:idx + size])
        idx += size
    return chunks


class ShardPlanner:
    def __init__(self, storage: Storage, config) -> None:
        self.storage = storage
        self.config = config

    def _seed_for(self, job: Job) -> int:
        # A job-stable seed: the same job definition always yields the same data
        # (useful for reproducible demos), yet different jobs differ.
        return (self.config.seed + sum(ord(c) for c in job.job_id)) % (2 ** 31 - 1)

    @staticmethod
    def _shape_warnings(strategy_name: str, records: list) -> list[str]:
        """Warn when the chosen strategy does not fit the data's size shape.

        Informational only — splitting still proceeds normally; the point is to
        make a strategy/shape mismatch visible on the shards page rather than
        silently producing unbalanced (or needlessly byte-balanced) shards.
        """
        if len(records) < 4:
            return []
        sizes = [record_size_bytes(r) for r in records[:5000]]
        mean = sum(sizes) / len(sizes)
        if mean <= 0:
            return []
        variance = sum((s - mean) ** 2 for s in sizes) / len(sizes)
        cv = (variance ** 0.5) / mean  # coefficient of variation of record size
        if cv >= 0.6 and strategy_name == "rows":
            return [
                f"记录大小差异较大（变异系数 {cv:.2f}），按行数均分会导致各分片数据量不均，"
                "建议改用「按数据量均衡」策略（record sizes are skewed; consider bytes strategy）"
            ]
        if cv < 0.1 and strategy_name == "bytes":
            return [
                f"记录大小基本一致（变异系数 {cv:.2f}），按数据量与按行数切分结果相同，"
                "按行数即可（uniform record sizes; rows strategy suffices）"
            ]
        return []

    # ------------------------------------------------------------------
    def plan(self, job: Job) -> dict:
        """Generate input records, split them into shards, and build tasks."""
        kind = job.params.get("input_kind", "wordcount")
        rows = max(1, int(job.input_rows))
        records = generate_input_records(kind, rows, self._seed_for(job))

        strategy_name = job.params.get("split_strategy") or DEFAULT_STRATEGY
        strategy, chunks, warnings = apply_strategy(strategy_name, records, job.num_map_tasks)
        warnings.extend(self._shape_warnings(strategy.name, records))
        strategy_name = strategy.name  # resolved name (unknown -> default)

        # Granularity: never create more map tasks than there are input records;
        # every produced shard owns exactly one map task.
        num_map = len(chunks)
        job.num_map_tasks = num_map

        input_shards: list[str] = []
        shard_meta: list[dict] = []
        total_bytes = 0
        total_records = 0
        for i, chunk in enumerate(chunks):
            sid = shard_id("in", i)
            count = len(chunk)
            size_bytes = sum(record_size_bytes(r) for r in chunk)
            total_bytes += size_bytes
            total_records += count
            self.storage.write({
                "shard_id": sid,
                "job_id": job.job_id,
                "stage": C.STAGE_INPUT,
                "index": i,
                "records": chunk,
                "count": count,
                "size_bytes": size_bytes,
                "created_ms": now_ms(),
            }, "jobs", job.job_id, "shards", C.STAGE_INPUT, f"{sid}.json")
            input_shards.append(sid)
            shard_meta.append({
                "shard_id": sid,
                "index": i,
                "count": count,
                "size_bytes": size_bytes,
            })

        # Durable manifest: the single source of truth the shards page reads.
        plan_doc = {
            "job_id": job.job_id,
            "strategy": strategy_name,
            "strategy_label": strategy.label,
            "size_basis": strategy.size_basis,
            "requested_shards": max(1, int(job.params.get("_requested_map_tasks", num_map) or num_map)),
            "actual_shards": num_map,
            "total_records": total_records,
            "total_bytes": total_bytes,
            "warnings": warnings,
            "shards": shard_meta,
            "created_ms": now_ms(),
        }
        self.storage.write(plan_doc, "jobs", job.job_id, "split_plan.json")

        map_tasks = [new_task(job, C.TASK_MAP, i) for i in range(num_map)]
        reduce_tasks = [new_task(job, C.TASK_REDUCE, p) for p in range(job.num_reduce_tasks)]

        return {
            "input_shards": input_shards,
            "map_tasks": map_tasks,
            "reduce_tasks": reduce_tasks,
            "total_records": total_records,
            "split_plan": plan_doc,
        }

    # ------------------------------------------------------------------
    def load_input_shard(self, job_id: str, shard: str) -> list[Any]:
        """Return exactly the records persisted for ``shard``.

        No slicing tricks: the shard page counts these same records, so what a
        map task processes is what the UI reports.
        """
        doc = self.storage.read("jobs", job_id, "shards", C.STAGE_INPUT, f"{shard}.json", default={})
        return doc.get("records", []) if doc else []

    def split_plan(self, job: Job) -> dict:
        """Read back the persisted split plan; derive it for legacy jobs."""
        doc = self.storage.read("jobs", job.job_id, "split_plan.json", default={})
        if doc:
            return doc
        # Jobs created before split plans existed: derive the view from the
        # shard files so they remain inspectable (their data is unchanged).
        shards = self.input_shards(job)
        strategy_name = job.params.get("split_strategy") or DEFAULT_STRATEGY
        strategy = get_strategy(strategy_name)
        return {
            "job_id": job.job_id,
            "strategy": strategy.name,
            "strategy_label": strategy.label,
            "size_basis": strategy.size_basis,
            "requested_shards": job.num_map_tasks,
            "actual_shards": len(shards),
            "total_records": sum(s.get("count", 0) for s in shards),
            "total_bytes": sum(s.get("size_bytes", 0) for s in shards),
            "warnings": [],
            "shards": shards,
            "created_ms": 0,
        }

    def input_shards(self, job: Job) -> list[dict]:
        out: list[dict] = []
        root = self.storage.path("jobs", job.job_id, "shards", C.STAGE_INPUT)
        for path in list_files(root, suffix=".json"):
            doc = read_json(path)
            if doc:
                records = doc.get("records")
                if records is None:
                    size = int(doc.get("size_bytes", -1))
                else:
                    # Recompute so legacy shard files (no size_bytes) still
                    # report a byte size identical to what the bytes strategy
                    # would compute.
                    size = sum(record_size_bytes(r) for r in records)
                out.append({
                    "shard_id": doc.get("shard_id"),
                    "index": doc.get("index", 0),
                    "count": doc.get("count", 0),
                    "size_bytes": size,
                    "stage": C.STAGE_INPUT,
                })
        return sorted(out, key=lambda d: d["index"])
