import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "engra_run_eval", Path(__file__).parent / "eval" / "run_eval.py"
)
run_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_eval)


def test_first_relevant_rank():
    assert run_eval.first_relevant_rank([False, False, True, False]) == 3
    assert run_eval.first_relevant_rank([True]) == 1
    assert run_eval.first_relevant_rank([False, False]) is None
    assert run_eval.first_relevant_rank([]) is None


def test_recall_at_k_hit_within_k():
    rel = [False, True, False]
    assert run_eval.recall_at_k(rel, 1) == 0.0
    assert run_eval.recall_at_k(rel, 2) == 1.0
    assert run_eval.recall_at_k(rel, 3) == 1.0


def test_recall_at_k_no_hit():
    assert run_eval.recall_at_k([False, False, False], 5) == 0.0


def test_recall_at_k_hit_beyond_k_is_miss():
    rel = [False, False, False, True]
    assert run_eval.recall_at_k(rel, 3) == 0.0
    assert run_eval.recall_at_k(rel, 4) == 1.0


def test_reciprocal_rank():
    assert run_eval.reciprocal_rank([True, False]) == 1.0
    assert run_eval.reciprocal_rank([False, True]) == 0.5
    assert run_eval.reciprocal_rank([False, False, False, True]) == 0.25
    assert run_eval.reciprocal_rank([False, False]) == 0.0


def test_mean():
    assert run_eval.mean([1.0, 0.0, 0.5]) == 0.5
    assert run_eval.mean([]) == 0.0


def test_aggregate():
    per_query = [
        [True, False, False],
        [False, False, True],
        [False, False, False],
    ]
    metrics = run_eval.aggregate(per_query, (1, 3))
    assert metrics["n"] == 3
    assert metrics["recall"][1] == 1.0 / 3
    assert metrics["recall"][3] == 2.0 / 3
    assert metrics["mrr"] == (1.0 + (1.0 / 3) + 0.0) / 3


def test_is_relevant_filename_and_page():
    record = {"expect_filename": "iso11783-5.pdf", "expect_page": 12}
    assert run_eval.is_relevant({"filename": "/corpus/iso11783-5.pdf", "page": 12}, record)
    assert not run_eval.is_relevant({"filename": "iso11783-5.pdf", "page": 9}, record)
    assert not run_eval.is_relevant({"filename": "iso11783-6.pdf", "page": 12}, record)


def test_is_relevant_page_agnostic_when_null():
    record = {"expect_filename": "iso11783-5.pdf", "expect_page": None}
    assert run_eval.is_relevant({"filename": "iso11783-5.pdf", "page": 99}, record)
    assert not run_eval.is_relevant({"filename": "other.pdf", "page": 99}, record)
