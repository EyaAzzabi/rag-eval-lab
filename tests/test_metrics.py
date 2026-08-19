"""Metrics are tested against cases worked out by hand.

If a metric is wrong, every number in the README is wrong and nothing else in the
test suite would catch it. These are the tests that matter most in this repo.
"""

import math

from rageval.metrics import (
    evaluate_run,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

RELEVANT = {"a": 1, "b": 1}


def test_recall_counts_against_total_relevant():
    assert recall_at_k(["a", "x", "y"], RELEVANT, 3) == 0.5
    assert recall_at_k(["a", "b", "y"], RELEVANT, 3) == 1.0
    assert recall_at_k(["x", "y", "z"], RELEVANT, 3) == 0.0


def test_recall_respects_the_cutoff():
    # "b" sits at rank 3 and must not count towards recall@2.
    assert recall_at_k(["a", "x", "b"], RELEVANT, 2) == 0.5


def test_precision_divides_by_k_not_by_relevant():
    assert precision_at_k(["a", "x", "y", "z"], RELEVANT, 4) == 0.25


def test_mrr_uses_the_first_hit_only():
    assert mrr_at_k(["x", "a", "b"], RELEVANT, 10) == 0.5
    assert mrr_at_k(["a", "x"], RELEVANT, 10) == 1.0
    assert mrr_at_k(["x", "y"], RELEVANT, 10) == 0.0


def test_ndcg_is_one_for_a_perfect_ranking():
    assert ndcg_at_k(["a", "b", "x"], RELEVANT, 3) == 1.0


def test_ndcg_penalises_a_lower_rank():
    # Single relevant document at rank 2: DCG = 1/log2(3), IDCG = 1.
    got = ndcg_at_k(["x", "a"], {"a": 1}, 2)
    assert math.isclose(got, 1 / math.log2(3), rel_tol=1e-9)


def test_ndcg_rewards_putting_the_higher_grade_first():
    graded = {"a": 2, "b": 1}
    assert ndcg_at_k(["a", "b"], graded, 2) > ndcg_at_k(["b", "a"], graded, 2)


def test_unjudged_queries_are_skipped_not_scored_zero():
    run = {"q1": ["a", "b"], "q_unjudged": ["x", "y"]}
    qrels = {"q1": {"a": 1, "b": 1}}
    scores = evaluate_run(run, qrels, ks=(2,))
    # Averaging over one judged query, not two, so a perfect run stays at 1.0.
    assert scores["recall@2"] == 1.0


def test_empty_relevant_set_is_zero_not_a_crash():
    assert recall_at_k(["a"], {}, 5) == 0.0
    assert ndcg_at_k(["a"], {}, 5) == 0.0
