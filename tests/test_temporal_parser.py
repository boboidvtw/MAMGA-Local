"""
Unit tests for memory/temporal_parser.py

Tests cover:
- Relative date expressions via extract_temporal_reference
- Absolute date parsing (multiple formats)
- Session timestamp parsing
- Temporal question detection
- Edge-cases: empty string, unknown formats
"""
import pytest
from datetime import datetime, timedelta

from memory.temporal_parser import TemporalParser


@pytest.fixture
def parser():
    return TemporalParser()


# ---------------------------------------------------------------------------
# Relative date resolution via extract_temporal_reference
# ---------------------------------------------------------------------------

class TestRelativeDates:
    def test_yesterday(self, parser, base_date):
        result = parser.extract_temporal_reference("yesterday", base_date)
        assert result == base_date - timedelta(days=1)

    def test_tomorrow(self, parser, base_date):
        result = parser.extract_temporal_reference("tomorrow", base_date)
        assert result == base_date + timedelta(days=1)

    def test_today(self, parser, base_date):
        result = parser.extract_temporal_reference("today", base_date)
        assert result == base_date

    def test_last_week(self, parser, base_date):
        result = parser.extract_temporal_reference("last week", base_date)
        assert result == base_date - timedelta(weeks=1)

    def test_next_week(self, parser, base_date):
        result = parser.extract_temporal_reference("next week", base_date)
        assert result == base_date + timedelta(weeks=1)

    def test_last_month(self, parser, base_date):
        result = parser.extract_temporal_reference("last month", base_date)
        assert result == base_date - timedelta(days=30)

    def test_last_year(self, parser, base_date):
        result = parser.extract_temporal_reference("last year", base_date)
        assert result == base_date - timedelta(days=365)

    def test_unknown_relative_returns_none(self, parser, base_date):
        result = parser.extract_temporal_reference("the day before the big bang", base_date)
        assert result is None

    def test_sentence_containing_relative(self, parser, base_date):
        """Works when the relative word is embedded in a sentence."""
        result = parser.extract_temporal_reference("We met yesterday for coffee", base_date)
        assert result is not None


# ---------------------------------------------------------------------------
# Session timestamp parsing
# ---------------------------------------------------------------------------

class TestSessionTimestamp:
    def test_locomo_format(self, parser):
        """Handle the LoComo dataset specific format: "1:56 pm on 8 May, 2023" """
        result = parser.parse_session_timestamp("1:56 pm on 8 May, 2023")
        assert result == datetime(2023, 5, 8)

    def test_iso_date(self, parser):
        result = parser.parse_session_timestamp("2023-06-15")
        assert result == datetime(2023, 6, 15)

    def test_iso_datetime(self, parser):
        result = parser.parse_session_timestamp("2023-06-15 09:30:00")
        assert result == datetime(2023, 6, 15, 9, 30, 0)

    def test_compact_datetime(self, parser):
        result = parser.parse_session_timestamp("202306150930")
        assert result.year == 2023
        assert result.month == 6

    def test_empty_string_returns_recent_datetime(self, parser):
        before = datetime.now()
        result = parser.parse_session_timestamp("")
        after = datetime.now()
        assert before <= result <= after


# ---------------------------------------------------------------------------
# Temporal question detection
# ---------------------------------------------------------------------------

class TestTemporalQuestionDetection:
    @pytest.mark.parametrize("question", [
        "When did Alice visit Paris?",
        "What time did the meeting start?",
        "How long ago did that happen?",
        "What date was the birthday party?",
    ])
    def test_temporal_questions_detected(self, parser, question):
        assert parser.is_temporal_question(question) is True

    @pytest.mark.parametrize("question", [
        "What is Alice's favourite colour?",
        "Who booked the hotel?",
        "How many people attended?",
    ])
    def test_non_temporal_questions_not_detected(self, parser, question):
        assert parser.is_temporal_question(question) is False
