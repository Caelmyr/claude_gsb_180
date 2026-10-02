"""Tests for deterministic hashing, shard planning and job submission."""

import shutil
import subprocess
import sys
import tempfile
import unittest

from backend.common import constants as C
from backend.common.config import ClusterConfig
from backend.common.hashing import partition_for, stable_hash
from backend.common.logbus import LogBus
from backend.common.storage import Storage
from backend.master.job_manager import JobManager
from backend.master.shard_planner import record_size, split_by_size, split_evenly


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

    def test_empty(self):
        chunks = split_evenly([], 4)
        self.assertEqual(len(chunks), 4)
        self.assertTrue(all(len(c) == 0 for c in chunks))

    def test_more_parts_than_items(self):
        chunks = split_evenly([1, 2], 10)
        self.assertEqual(sum(len(c) for c in chunks), 2)


class TestRecordSize(unittest.TestCase):
    def test_text_record(self):
        self.assertEqual(record_size("hello"), 5)
        self.assertEqual(record_size("中文"), 6)  # 3 bytes per char in UTF-8

    def test_structured_record(self):
        self.assertGreater(record_size({"key": "alpha", "value": 42}), 0)

    def test_unmeasurable(self):
        self.assertEqual(record_size(None), 0)


class TestSplitBySize(unittest.TestCase):
    def test_equal_sizes_match_count_split(self):
        items = ["x"] * 10
        chunks = split_by_size(items, [1] * 10, 3)
        self.assertEqual(sorted(len(c) for c in chunks), [3, 3, 4])

    def test_balances_bytes_not_counts(self):
        # 2 big records (10 B) + 4 small ones (1 B): total 24, target 12.
        # Count split would give byte sizes [21, 3]; size split gives [10, 14].
        items = ["b" * 10, "b" * 10, "s", "s", "s", "s"]
        sizes = [record_size(i) for i in items]
        chunks = split_by_size(items, sizes, 2)
        shard_bytes = [sum(record_size(r) for r in c) for c in chunks]
        self.assertEqual(shard_bytes, [10, 14])
        self.assertEqual([len(c) for c in chunks], [1, 5])

    def test_contiguous_order_and_total_preserved(self):
        items = [f"rec-{i}-{'x' * (i % 7)}" for i in range(50)]
        sizes = [record_size(i) for i in items]
        chunks = split_by_size(items, sizes, 6)
        self.assertEqual([r for c in chunks for r in c], items)

    def test_never_empty_while_items_remain(self):
        items = ["x"] * 5
        chunks = split_by_size(items, [1] * 5, 5)
        self.assertEqual(len(chunks), 5)
        self.assertTrue(all(len(c) == 1 for c in chunks))

    def test_more_parts_than_items_clamps(self):
        chunks = split_by_size(["a", "b"], [1, 1], 10)
        self.assertEqual(sum(len(c) for c in chunks), 2)
        self.assertTrue(all(len(c) > 0 for c in chunks))

    def test_huge_record_yields_fewer_shards_but_none_empty(self):
        items = ["h" * 100, "a", "b", "c"]
        sizes = [record_size(i) for i in items]
        chunks = split_by_size(items, sizes, 4)
        self.assertLess(len(chunks), 4)
        self.assertTrue(all(len(c) > 0 for c in chunks))
        self.assertEqual([r for c in chunks for r in c], items)

    def test_zero_sizes_fall_back_to_count_split(self):
        items = [None, None, None, None]
        chunks = split_by_size(items, [0, 0, 0, 0], 2)
        self.assertEqual([len(c) for c in chunks], [2, 2])

    def test_empty_input(self):
        chunks = split_by_size([], [], 3)
        self.assertEqual(sum(len(c) for c in chunks), 0)


class TestSubmit(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.storage = Storage(self.tmp)
        self.jm = JobManager(self.storage, ClusterConfig(), LogBus(self.storage))

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
        # persisted round-trip
        self.assertEqual(self.jm.get_job(job.job_id).name, "t")

    def test_submit_rejects_unknown_mapper(self):
        with self.assertRaises(ValueError):
            self.jm.submit({"name": "t", "mapper": "nope", "reducer": "count_reducer"})

    def test_granularity_clamps_map_tasks(self):
        job = self.jm.submit({
            "name": "t", "mapper": "wordcount_mapper", "reducer": "count_reducer",
            "num_map_tasks": 100, "num_reduce_tasks": 2, "input_rows": 30, "params": {},
        })
        # never more map tasks than input records
        self.assertLessEqual(len(job.map_task_ids), 30)


class TestSplitStrategySubmit(unittest.TestCase):
    """End-to-end: submit with a split strategy, then verify the persisted
    shards match exactly what the job will process."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.storage = Storage(self.tmp)
        self.jm = JobManager(self.storage, ClusterConfig(), LogBus(self.storage))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _submit(self, **extra):
        payload = {
            "name": "t", "mapper": "wordcount_mapper", "reducer": "count_reducer",
            "num_map_tasks": 4, "num_reduce_tasks": 2, "input_rows": 1000, "params": {},
        }
        payload.update(extra)
        return self.jm.submit(payload)

    def _assert_display_matches_processing(self, job):
        """The shards page (persisted docs) and the map-task input must agree."""
        shards = self.jm.planner.input_shards(job)
        self.assertEqual(len(shards), job.num_map_tasks)
        self.assertEqual(len(shards), len(self.jm.tasks_for(job.job_id, C.TASK_MAP)))
        total = 0
        for s in shards:
            records = self.jm.planner.load_input_shard(job.job_id, s["shard_id"])
            # what the map task will consume == what the page displays
            self.assertEqual(len(records), s["count"])
            self.assertGreater(s["count"], 0)  # no empty shards
            total += s["count"]
        self.assertEqual(total, job.stats["total_records"])
        self.assertEqual(total, job.input_rows)
        return shards

    def test_default_strategy_is_count(self):
        job = self._submit()
        self.assertEqual(job.split_strategy, C.SPLIT_BY_COUNT)
        self.assertEqual(job.stats["split_strategy"], C.SPLIT_BY_COUNT)

    def test_count_strategy_uneven_rows(self):
        # 1000 rows / 6 shards: not divisible -> sizes differ by at most one.
        job = self._submit(num_map_tasks=6)
        shards = self._assert_display_matches_processing(job)
        counts = [s["count"] for s in shards]
        self.assertEqual(sum(counts), 1000)
        self.assertLessEqual(max(counts) - min(counts), 1)

    def test_size_strategy_records_bytes(self):
        job = self._submit(split_strategy="size", num_map_tasks=4)
        self.assertEqual(job.split_strategy, C.SPLIT_BY_SIZE)
        shards = self._assert_display_matches_processing(job)
        for s in shards:
            self.assertIsNotNone(s["bytes"])
            self.assertGreater(s["bytes"], 0)
        self.assertEqual(sum(s["bytes"] for s in shards), job.stats["total_bytes"])
        # generated lines are similar in length, so 4 shards stay balanced
        counts = [s["count"] for s in shards]
        self.assertLessEqual(max(counts) - min(counts), max(2, min(counts)))

    def test_size_strategy_kv_records(self):
        job = self.jm.submit({
            "name": "t", "mapper": "kv_mapper", "reducer": "sum_reducer",
            "num_map_tasks": 3, "num_reduce_tasks": 2, "input_rows": 300,
            "split_strategy": "size", "params": {},
        })
        shards = self._assert_display_matches_processing(job)
        self.assertEqual(sum(s["count"] for s in shards), 300)

    def test_unknown_strategy_rejected(self):
        with self.assertRaises(ValueError):
            self._submit(split_strategy="random")

    def test_strategy_persisted_per_job(self):
        # Switching strategy between submissions must not leak across jobs.
        job_a = self._submit(name="a", split_strategy="count")
        job_b = self._submit(name="b", split_strategy="size")
        self.assertEqual(self.jm.get_job(job_a.job_id).split_strategy, C.SPLIT_BY_COUNT)
        self.assertEqual(self.jm.get_job(job_b.job_id).split_strategy, C.SPLIT_BY_SIZE)
        shards_a = self.jm.planner.input_shards(job_a)
        shards_b = self.jm.planner.input_shards(job_b)
        self.assertTrue(all(s["split_strategy"] == C.SPLIT_BY_COUNT for s in shards_a))
        self.assertTrue(all(s["split_strategy"] == C.SPLIT_BY_SIZE for s in shards_b))

    def test_single_record_single_shard(self):
        job = self._submit(input_rows=1, num_map_tasks=8)
        shards = self._assert_display_matches_processing(job)
        self.assertEqual(len(shards), 1)
        self.assertEqual(shards[0]["count"], 1)


if __name__ == "__main__":
    unittest.main()
