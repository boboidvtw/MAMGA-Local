"""
Unit tests for memory/keyword_enrichment.py

Tests cover:
- Named entity extraction
- Year / month / date extraction
- Stop-word filtering
- Bigram extraction
- Content enrichment
- Edge-cases: empty input, all stop-words, long text
"""
import pytest

from memory.keyword_enrichment import KeywordEnricher


@pytest.fixture
def enricher():
    return KeywordEnricher()


# ---------------------------------------------------------------------------
# extract_keywords
# ---------------------------------------------------------------------------

class TestExtractKeywords:
    def test_extracts_named_entities(self, enricher):
        text = "Alice and Bob travelled to Paris last summer."
        keywords = enricher.extract_keywords(text)
        assert any(k in ("alice", "bob", "paris") for k in keywords)

    def test_extracts_full_year(self, enricher):
        """Year should be returned as 4-digit string, not just the century prefix."""
        text = "The event happened in 2023."
        keywords = enricher.extract_keywords(text)
        assert "2023" in keywords
        assert "20" not in keywords  # confirm capture-group bug is fixed

    def test_extracts_month(self, enricher):
        text = "We met in January."
        keywords = enricher.extract_keywords(text)
        assert "january" in keywords

    def test_stop_words_excluded(self, enricher):
        """Common stop-words should not appear in keywords."""
        text = "The cat sat on the mat and it was very good."
        keywords = enricher.extract_keywords(text)
        for sw in ("the", "and", "it", "was", "very", "on"):
            assert sw not in keywords

    def test_empty_text_returns_empty_list(self, enricher):
        assert enricher.extract_keywords("") == []

    def test_respects_max_keywords(self, enricher):
        text = " ".join(f"word{i}" for i in range(50))
        keywords = enricher.extract_keywords(text, max_keywords=5)
        assert len(keywords) <= 5

    def test_all_stop_words_returns_empty_or_minimal(self, enricher):
        text = "the and or but if is was"
        keywords = enricher.extract_keywords(text)
        # None of these should appear
        for word in text.split():
            assert word not in keywords


# ---------------------------------------------------------------------------
# extract_bigrams
# ---------------------------------------------------------------------------

class TestExtractBigrams:
    def test_bigram_extracted(self, enricher):
        text = "machine learning is a subfield of artificial intelligence"
        bigrams = enricher.extract_bigrams(text)
        # At least one meaningful bigram expected
        assert len(bigrams) > 0

    def test_bigrams_are_strings(self, enricher):
        bigrams = enricher.extract_bigrams("alice visited paris last monday")
        for b in bigrams:
            assert isinstance(b, str)
            # bigrams are stored as "word1_word2" (underscore-separated)
            assert "_" in b or " " in b  # accept either separator


# ---------------------------------------------------------------------------
# enrich_content
# ---------------------------------------------------------------------------

class TestEnrichContent:
    def test_enrich_returns_string(self, enricher):
        result = enricher.enrich_content("Alice went to the market.")
        assert isinstance(result, str)

    def test_enrich_contains_original_text(self, enricher):
        text = "Alice went to the market."
        result = enricher.enrich_content(text)
        assert "alice" in result.lower() or "market" in result.lower()

    def test_enrich_empty_text(self, enricher):
        result = enricher.enrich_content("")
        assert isinstance(result, str)
