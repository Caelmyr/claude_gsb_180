"""Tests for deterministic hashing, shard planning and job submission."""

import shutil
import subprocess
import sys
import tempfile
import unittest

from backend.common.config import ClusterConfig
from backend.common.hashing import partition_for, stable_hash
from backend.common.logbus import LogBus
from backend.common.storage import Storage
from backend.master.job_manager import JobManager
from backend.master.shard_planner import ShardPlanner, split_evenly
from backend.master.split_strategy import (
    apply_strategy,
    record_size_bytes,
    split_by_bytes,
    split_by_rows,
)
from backend.tasks.samples import generate_input_records


def make_jm(tmp):
    storage = Storage(tmp)
    return storage, JobManager(storage, ClusterConfig(), LogBus(storage))


class TestHashing(unittest.TestCase):
    def test_deterministic(self):
        self.assertEqual(partition_for("the", 4), partition_for("the", 4))
        self.assertEqual(stable_hash("key"), stable_hash("key"))

    def test_in_range(self):
        for i in range(200):
            self.assertTrue(0 <= partition_for(f"k{i}", 7) < 7)

    def test_stable_across_processes(self):
        code = "from backend.common.hashing import partition_for; print(partition_for('word', 8))"
        out = subprocess.check_output([sys.executable, "-c", code], text=True).strip()
        self.assertEqual(int(out), partition_for("word", 8))


class TestSplitEvenly(unittest.TestCase):
    def test_sizes_differ_by_at_most_one(self):
        chunks = split_evenly(list(range(10)), 3)
        self.assertEqual(sorted(len(c) for c in chunks), [3, 3, 4])

    def test_non_divisible_remainder_is_spread(self):
        chunks = split_evenly(list(range(10)), 3)
        # remainder 2 -> first two shards take the extra record
        self.assertEqual([len(c) for c in chunks], [4, 3, 3])
        # no tiny/empty trailing shard
        self.assertEqual(len(chunks[-1]), 3)

    def test_empty(self):
        # empty input: exactly one empty shard, never many empty tasks
        chunks = split_evenly([], 4)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks, [[]])

    def test_more_parts_than_items(self):
        chunks = split_evenly([1, 2], 10)
        self.assertEqual([len(c) for c in chunks], [1, 1])
        self.assertEqual(sum(len(c) for c in chunks), 2)


class TestSplitStrategies(unittest.TestCase):
    def test_rows_clamps_and_warns(self):
        chunks, warnings = split_by_rows([1, 2], 10)
        self.assertEqual(len(chunks), 2)
        self.assertTrue(all(len(c) == 1 for c in chunks))
        self.assertTrue(warnings)

    def test_rows_empty_single_shard(self):
        chunks, warnings = split_by_rows([], 5)
        self.assertEqual(chunks, [[]])
        self.assertEqual(len(chunks), 1)
        self.assertTrue(warnings)

    def test_rows_covers_every_record_exactly_once(self):
        records = list(range(100))
        chunks, _ = split_by_rows(records, 7)
        flat = [r for c in chunks for r in c]
        self.assertEqual(flat, records)  # contiguous, no loss, no duplication

    def test_bytes_balances_skewed_sizes(self):
        records = ["x", "x", "x", "x", "y" * 1000, "x", "x"]
        chunks, _ = split_by_bytes(records, 3)
        sizes = [sum(record_size_bytes(r) for r in c) for c in chunks]
        self.assertEqual(len(chunks), 3)
        # every record is present exactly once and stays whole
        self.assertEqual(sum(len(c) for c in chunks), len(records))
        from collections import Counter
        self.assertEqual(Counter(r for c in chunks for r in c), Counter(records))
        # the huge record dominates one shard; no shard is empty
        self.assertTrue(all(s > 0 for s in sizes))

    def test_bytes_oversized_record_warns_not_split(self):
        records = ["z" * 5000, "a", "b"]
        chunks, warnings = split_by_bytes(records, 3)
        self.assertTrue(any(("z" * 5000) in c for c in chunks))
        # oversized single record is kept whole (one shard contains it alone)
        self.assertTrue(any(len(c) == 1 and c[0] == "z" * 5000 for c in chunks))

    def test_bytes_empty(self):
        chunks, _ = split_by_bytes([], 4)
        self.assertEqual(chunks, [[]])

    def test_bytes_requested_more_than_records(self):
        records = ["a", "bb", "ccc"]
        chunks, warnings = split_by_bytes(records, 10)
        self.assertEqual(len(chunks), 3)
        self.assertTrue(all(len(c) == 1 for c in chunks))
        self.assertTrue(warnings)

    def test_unknown_strategy_falls_back(self):
        records = ["a", "b", "c"]
        strategy, chunks, _ = apply_strategy("does-not-exist", records, 2)
        self.assertEqual(strategy.name, "rows")
        self.assertEqual([len(c) for c in chunks], [2, 1])


class TestShapeWarnings(unittest.TestCase):
    def _warnings(self, strategy, records):
        tmp = tempfile.mkdtemp()
        try:
            storage = Storage(tmp)
            planner = ShardPlanner(storage, ClusterConfig())
            return planner._shape_warnings(strategy, records)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_rows_warns_on_skewed_sizes(self):
        records = ["x"] * 20 + ["word " * 200] * 20
        warnings = self._warnings("rows", records)
        self.assertTrue(warnings)

    def test_bytes_warns_on_uniform_sizes(self):
        records = ["abcdef"] * 40
        warnings = self._warnings("bytes", records)
        self.assertTrue(warnings)

    def test_no_warning_when_strategy_fits(self):
        records = ["x"] * 20 + ["word " * 200] * 20
        self.assertEqual(self._warnings("bytes", records), [])


class TestSubmit(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.storage, self.jm = make_jm(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_submit_creates_shards_and_tasks(self):
        job = self.jm.submit({
            "name": "t", "mapper": "wordcount_mapper", "reducer": "count_reducer",
            "num_map_tasks": 4, "num_reduce_tasks": 2, "input_rows": 800, "params": {},
        })
        self.assertEqual(job.status, "MAP")
        self.assertEqual(len(job.map_task_ids), 4)
        self.assertEqual(len(job.reduce_task_ids), 2)
        self.assertEqual(len(self.jm.tasks_for(job.job_id)), 6)
        self.assertEqual(job.split_strategy, "rows")
        self.assertEqual(self.jm.get_job(job.job_id).name, "t")

    def test_submit_rejects_unknown_mapper(self):
        with self.assertRaises(ValueError):
            self.jm.submit({"name": "t", "mapper": "nope", "reducer": "count_reducer"})

    def test_submit_rejects_unknown_strategy(self):
        with self.assertRaises(ValueError):
            self.jm.submit({
                "name": "t", "mapper": "wordcount_mapper", "reducer": "count_reducer",
                "num_map_tasks": 4, "num_reduce_tasks": 2, "input_rows": 100,
                "split_strategy": "by-vibes",
            })

    def test_granularity_clamps_map_tasks(self):
        job = self.jm.submit({
            "name": "t", "mapper": "wordcount_mapper", "reducer": "count_reducer",
            "num_map_tasks": 100, "num_reduce_tasks": 2, "input_rows": 30, "params": {},
        })
        # never more map tasks than input records
        self.assertLessEqual(len(job.map_task_ids), 30)
        self.assertEqual(job.stats["split"]["actual_shards"], len(job.map_task_ids))
        self.assertTrue(job.stats["split"]["warnings"])

    def test_input_rows_matches_generated_records(self):
        # the generator must produce exactly input_rows records (no off-by-one)
        rows = 123
        records = generate_input_records("wordcount", rows, seed=7)
        self.assertEqual(len(records), rows)

    def test_displayed_shard_counts_equal_records_workers_process(self):
        # Core consistency requirement: shards page view == dispatched records.
        job = self.jm.submit({
            "name": "t", "mapper": "wordcount_mapper", "reducer": "count_reducer",
            "num_map_tasks": 3, "num_reduce_tasks": 2, "input_rows": 100, "params": {},
        })
        planner = self.jm.planner
        view = planner.input_shards(job)
        plan = planner.split_plan(job)
        total_view = 0
        for meta in view:
            shard = meta["shard_id"]
            dispatched = planner.load_input_shard(job.job_id, shard)
            # the map task receives exactly as many records as the page shows
            self.assertEqual(len(dispatched), meta["count"])
            # and the byte size shown equals the actual serialized payload
            self.assertEqual(sum(record_size_bytes(r) for r in dispatched),
                             meta["size_bytes"])
            total_view += meta["count"]
        self.assertEqual(total_view, 100)
        self.assertEqual(plan["total_records"], 100)
        self.assertEqual(job.stats["total_records"], 100)

    def test_bytes_strategy_non_divisible_rows(self):
        # 100 rows into 6 shards under each strategy: coverage + no empty shard
        for strategy in ("rows", "bytes"):
            job = self.jm.submit({
                "name": "t", "mapper": "wordcount_mapper", "reducer": "count_reducer",
                "num_map_tasks": 6, "num_reduce_tasks": 1, "input_rows": 100,
                "split_strategy": strategy,
            })
            view = self.jm.planner.input_shards(job)
            self.assertEqual(len(view), 6, strategy)
            self.assertTrue(all(s["count"] > 0 for s in view), strategy)
            self.assertEqual(sum(s["count"] for s in view), 100, strategy)
            self.assertEqual(job.params["split_strategy"], strategy)

    def test_split_plan_persisted_and_survives_reload(self):
        storage, jm = make_jm(self.tmp)
        job = jm.submit({
            "name": "t", "mapper": "wordcount_mapper", "reducer": "count_reducer",
            "num_map_tasks": 4, "num_reduce_tasks": 1, "input_rows": 50,
            "split_strategy": "bytes",
        })
        plan_doc = storage.read("jobs", job.job_id, "split_plan.json")
        self.assertEqual(plan_doc["strategy"], "bytes")
        self.assertEqual(plan_doc["actual_shards"], 4)
        # reloaded manager reads same locked strategy
        jm2 = JobManager(storage, ClusterConfig(), LogBus(storage))
        job2 = jm2.get_job(job.job_id)
        self.assertEqual(job2.split_strategy, "bytes")
        self.assertEqual(len(jm2.planner.input_shards(job2)), 4)


if __name__ == "__main__":
    unittest.main()
