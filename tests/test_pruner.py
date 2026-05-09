"""Tests for Svalinn SWEPruner."""

import pytest
from svalinn import SWEPruner


@pytest.fixture
def pruner() -> SWEPruner:
    return SWEPruner()


# ── token_count ────────────────────────────────────────────────────────

class TestTokenCount:
    def test_empty_string(self, pruner):
        assert pruner.token_count("") == 0

    def test_single_word(self, pruner):
        # 1 word × 1.3 ratio ≈ 2 (ceiled)
        assert pruner.token_count("hello") == 2

    def test_sentence(self, pruner):
        tokens = pruner.token_count("The quick brown fox jumps over the lazy dog")
        # 9 words × 1.3 ≈ 12
        assert tokens == 12

    def test_custom_ratio(self):
        p = SWEPruner(word_to_token_ratio=1.0)
        assert p.token_count("one two three") == 3


# ── prune ──────────────────────────────────────────────────────────────

class TestPrune:
    def test_short_context_passes_through(self, pruner):
        text = "Hello world."
        result = pruner.prune(text, max_tokens=1000)
        assert result == text

    def test_empty_context(self, pruner):
        assert pruner.prune("", 100) == ""

    def test_deduplication(self, pruner):
        # Use a low max_tokens so dedup step actually triggers
        text = "Alpha block.\n\nAlpha block."
        result = pruner.prune(text, max_tokens=5)
        assert result.count("Alpha block.") == 1

    def test_prune_reduces_tokens(self, pruner):
        long_text = ". ".join(f"Sentence number {i} about topic" for i in range(50))
        original_tokens = pruner.token_count(long_text)
        result = pruner.prune(long_text, max_tokens=original_tokens // 2)
        assert pruner.token_count(result) <= original_tokens // 2 + 20  # tolerance

    def test_prune_preserves_important_content(self, pruner):
        text = "Critical fact: the server crashed on 2024-01-15. " * 20
        result = pruner.prune(text, max_tokens=50)
        assert "Critical" in result or "2024" in result


# ── summarize ──────────────────────────────────────────────────────────

class TestSummarize:
    def test_single_sentence_returns_itself(self, pruner):
        assert pruner.summarize("Just one sentence here.", ratio=0.5) == "Just one sentence here."

    def test_summarize_reduces_length(self, pruner):
        text = ". ".join(f"Sentence number {i} talks about something interesting" for i in range(10))
        result = pruner.summarize(text, ratio=0.3)
        assert len(result.split()) < len(text.split())

    def test_summarize_empty(self, pruner):
        assert pruner.summarize("", ratio=0.5) == ""

    def test_ratio_clamped(self, pruner):
        text = "First sentence. Second sentence. Third sentence."
        # ratio > 1.0 should be clamped to 1.0
        result = pruner.summarize(text, ratio=2.0)
        assert "First" in result


# ── extract_key_facts ──────────────────────────────────────────────────

class TestExtractKeyFacts:
    def test_extracts_facts_with_numbers(self, pruner):
        text = "The population of Reykjavik is 131,000."
        facts = pruner.extract_key_facts(text)
        assert len(facts) >= 1
        assert any("131" in f.content for f in facts)

    def test_extracts_definition_patterns(self, pruner):
        text = "Svalinn is a mythical shield in Norse mythology."
        facts = pruner.extract_key_facts(text)
        assert len(facts) >= 1
        assert "Svalinn" in facts[0].content or "shield" in facts[0].content

    def test_empty_context(self, pruner):
        assert pruner.extract_key_facts("") == []

    def test_max_facts_limit(self, pruner):
        text = "\n".join(f"Fact number {i} is about something important on date 2024-01-{i:02d}." for i in range(1, 20))
        facts = pruner.extract_key_facts(text, max_facts=5)
        assert len(facts) <= 5

    def test_fact_importance_scored(self, pruner):
        text = "An uninteresting thing happened. The GDP grew 3.2% in Q4 2024."
        facts = pruner.extract_key_facts(text)
        if len(facts) >= 2:
            # GDP sentence should score higher
            gdp_fact = next((f for f in facts if "GDP" in f.content), None)
            boring_fact = next((f for f in facts if "uninteresting" in f.content), None)
            if gdp_fact and boring_fact:
                assert gdp_fact.importance >= boring_fact.importance