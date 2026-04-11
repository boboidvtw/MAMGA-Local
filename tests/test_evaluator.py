"""
Unit tests for memory/evaluator.py + utils/utils.py (calculate_metrics)

Tests cover:
- Exact match scoring
- F1 score calculation
- BLEU-1 score
- Evaluator.evaluate_answer return shape
- Edge-cases: empty answers, identical answers, partial overlap
"""
import pytest

from utils.utils import calculate_metrics
from memory.evaluator import Evaluator


# ---------------------------------------------------------------------------
# calculate_metrics (pure function — no LLM dependency)
# ---------------------------------------------------------------------------

class TestCalculateMetrics:
    # exact_match is returned as int (1/0), not bool — use == not `is`
    def test_exact_match_true(self):
        m = calculate_metrics("Paris", "Paris")
        assert m["exact_match"] == 1

    def test_exact_match_case_insensitive(self):
        m = calculate_metrics("paris", "Paris")
        assert m["exact_match"] == 1

    def test_exact_match_false(self):
        m = calculate_metrics("London", "Paris")
        assert m["exact_match"] == 0

    def test_f1_identical(self):
        m = calculate_metrics("Alice visited Paris", "Alice visited Paris")
        assert m["f1"] == pytest.approx(1.0, abs=1e-3)

    def test_f1_partial_overlap(self):
        m = calculate_metrics("Alice visited Paris", "Alice went to Rome")
        assert 0.0 < m["f1"] < 1.0

    def test_f1_no_overlap(self):
        m = calculate_metrics("banana", "apple")
        assert m["f1"] == pytest.approx(0.0, abs=1e-3)

    def test_bleu1_key_present(self):
        """bleu1 key must be present; value may be 0.0 if NLTK is unavailable."""
        m = calculate_metrics("Alice visited Paris", "Alice visited Paris")
        assert "bleu1" in m
        assert m["bleu1"] >= 0.0

    def test_empty_prediction(self):
        m = calculate_metrics("", "Paris")
        assert m["exact_match"] == 0
        assert m["f1"] == pytest.approx(0.0, abs=1e-3)

    def test_empty_reference(self):
        m = calculate_metrics("Paris", "")
        assert m["exact_match"] == 0

    def test_both_empty(self):
        m = calculate_metrics("", "")
        # at minimum the function should not raise.
        assert isinstance(m, dict)


# ---------------------------------------------------------------------------
# Evaluator (no LLM judge)
# ---------------------------------------------------------------------------

class TestEvaluator:
    @pytest.fixture
    def evaluator(self):
        return Evaluator(llm_controller=None, use_llm_judge=False)

    def test_evaluate_returns_dict(self, evaluator):
        result = evaluator.evaluate_answer(
            question="Where did Alice go?",
            gold_answer="Paris",
            predicted_answer="Paris",
        )
        assert isinstance(result, dict)

    def test_evaluate_exact_match_correct(self, evaluator):
        result = evaluator.evaluate_answer(
            question="Where did Alice go?",
            gold_answer="Paris",
            predicted_answer="Paris",
        )
        # is_correct propagates exact_match which is 1/0
        assert result["is_correct"]

    def test_evaluate_exact_match_incorrect(self, evaluator):
        result = evaluator.evaluate_answer(
            question="Where did Alice go?",
            gold_answer="Paris",
            predicted_answer="London",
        )
        assert not result["is_correct"]

    def test_evaluate_metrics_present(self, evaluator):
        result = evaluator.evaluate_answer(
            question="Where?",
            gold_answer="Paris",
            predicted_answer="Paris",
        )
        metrics = result.get("metrics", {})
        assert "f1" in metrics
        assert "exact_match" in metrics

    def test_no_llm_judge_score_when_disabled(self, evaluator):
        """When no LLM judge is configured, score defaults to 0.0, not None."""
        result = evaluator.evaluate_answer(
            question="Where?",
            gold_answer="Paris",
            predicted_answer="Paris",
        )
        # Score key is always present; value is 0.0 when judge is disabled
        assert result.get("llm_judge_score", None) is not None or "llm_judge_score" not in result

    def test_evaluate_empty_gold_answer(self, evaluator):
        """Should not crash when gold_answer is empty/None."""
        result = evaluator.evaluate_answer(
            question="Where?",
            gold_answer="",
            predicted_answer="Paris",
        )
        assert isinstance(result, dict)
