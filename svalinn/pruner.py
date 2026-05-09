"""SWEPruner — core context pruning, summarization, and key-fact extraction."""

from __future__ import annotations

import re
import math
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class KeyFact:
    """A single key fact extracted from a context."""
    content: str
    source_line: int
    importance: float  # 0.0–1.0


class SWEPruner:
    """Shield before the sun. Prunes and summarizes long contexts for LLM consumption.

    Svalinn is the mythical shield that stands before the sun. Similarly,
    SWEPruner shields your context window from being consumed by unnecessary
    tokens, keeping only what matters.

    Methods
    -------
    prune(context, max_tokens)
        Reduce context to fit within a token budget.
    summarize(context, ratio)
        Produce a shorter summary of the context.
    extract_key_facts(context)
        Pull out the most important facts from the context.
    token_count(text)
        Count approximate tokens in text.
    """

    # Rough heuristic: average English word ≈ 1.3 tokens
    WORD_TO_TOKEN_RATIO = 1.3
    SENTENCE_SPLITTER = re.compile(r"(?<=[.!?])\s+")
    PARAGRAPH_SPLITTER = re.compile(r"\n\s*\n")
    LINE_SPLITTER = re.compile(r"\n")
    HEADER_PATTERN = re.compile(r"^(#{1,6}\s|[-*]\s|\d+\.\s)", re.MULTILINE)

    def __init__(self, *, word_to_token_ratio: float = 1.3):
        self.WORD_TO_TOKEN_RATIO = word_to_token_ratio

    # ------------------------------------------------------------------
    # Token estimation
    # ------------------------------------------------------------------
    def token_count(self, text: str) -> int:
        """Estimate the number of tokens in *text*.

        Uses a simple heuristic based on word count. Good enough for
        budget planning; not a replacement for a real tokenizer.
        """
        if not text:
            return 0
        words = len(text.split())
        return math.ceil(words * self.WORD_TO_TOKEN_RATIO)

    # ------------------------------------------------------------------
    # Pruning
    # ------------------------------------------------------------------
    def prune(self, context: str, max_tokens: int) -> str:
        """Prune *context* to fit within *max_tokens*.

        Strategy (applied in order until budget is met):
        1. Remove duplicate paragraphs (keep first occurrence).
        2. Strip lines that look like boilerplate (e.g., repeated headers).
        3. Remove least-important sentences from the tail until we fit.
        """
        if not context:
            return ""

        # Step 0 — early return if already within budget
        if self.token_count(context) <= max_tokens:
            return context

        # Step 1 — deduplicate paragraphs
        seen: set[str] = set()
        unique_paragraphs: list[str] = []
        for para in self.PARAGRAPH_SPLITTER.split(context):
            normalised = para.strip().lower()
            if normalised not in seen:
                seen.add(normalised)
                unique_paragraphs.append(para)

        text = "\n\n".join(unique_paragraphs)
        if self.token_count(text) <= max_tokens:
            return text

        # Step 2 — strip boilerplate / header-only lines
        lines = text.split("\n")
        filtered_lines: list[str] = []
        for line in lines:
            stripped = line.strip()
            if stripped and not (self.HEADER_PATTERN.match(stripped) and len(stripped.split()) <= 5):
                filtered_lines.append(line)
            elif not stripped:
                filtered_lines.append(line)

        text = "\n".join(filtered_lines)
        if self.token_count(text) <= max_tokens:
            return text

        # Step 3 — sentence-level pruning (remove least important first)
        sentences = self._split_sentences(text)
        indexed = list(enumerate(sentences))
        ranked = sorted(indexed, key=lambda s: len(s[1].split()), reverse=True)

        total_tokens = self.token_count(text)
        removed_indices: set[int] = set()

        for idx, sent in ranked:
            if total_tokens <= max_tokens:
                break
            sent_tokens = self.token_count(sent)
            total_tokens -= sent_tokens
            removed_indices.add(idx)

        result = " ".join(s for i, s in enumerate(sentences) if i not in removed_indices)
        return result.strip()

    # ------------------------------------------------------------------
    # Summarization
    # ------------------------------------------------------------------
    def summarize(self, context: str, ratio: float = 0.5) -> str:
        """Summarize *context* to approximately *ratio* of its original length.

        Uses an extractive approach: score sentences by word-frequency
        importance and keep the top-scoring ones in original order.
        """
        if not context:
            return ""
        ratio = max(0.0, min(ratio, 1.0))

        sentences = self._split_sentences(context)
        if len(sentences) <= 1:
            return context

        target_count = max(1, int(len(sentences) * ratio))

        # Word frequency scoring
        word_freq: dict[str, int] = {}
        for sent in sentences:
            for word in sent.lower().split():
                word = re.sub(r"[^\w]", "", word)
                if len(word) > 2:
                    word_freq[word] = word_freq.get(word, 0) + 1

        # Score each sentence
        scores: list[Tuple[int, float]] = []
        for i, sent in enumerate(sentences):
            score = 0.0
            words = sent.lower().split()
            for w in words:
                w = re.sub(r"[^\w]", "", w)
                if w in word_freq:
                    score += word_freq[w]
            # Normalise by sentence length to avoid favouring long sentences
            score = score / max(1, len(words))
            # Boost first sentence (lead sentence often most important)
            if i == 0:
                score *= 1.5
            scores.append((i, score))

        # Pick top sentences by score
        scores.sort(key=lambda x: x[1], reverse=True)
        selected_indices = sorted([idx for idx, _ in scores[:target_count]])

        return " ".join(sentences[i] for i in selected_indices).strip()

    # ------------------------------------------------------------------
    # Key fact extraction
    # ------------------------------------------------------------------
    def extract_key_facts(self, context: str, max_facts: int = 10) -> List[KeyFact]:
        """Extract up to *max_facts* key facts from *context*.

        Heuristically identifies sentences that contain factual signals
        (numbers, proper nouns, dates, definitions).
        """
        if not context:
            return []

        lines = context.split("\n")
        all_facts: list[KeyFact] = []

        for line_num, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            sentences = self._split_sentences(line)
            for sent in sentences:
                importance = self._fact_importance(sent)
                if importance > 0.25:
                    all_facts.append(KeyFact(
                        content=sent.strip(),
                        source_line=line_num,
                        importance=round(importance, 3),
                    ))

        # Sort by importance, keep top max_facts
        all_facts.sort(key=lambda f: f.importance, reverse=True)
        return all_facts[:max_facts]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split *text* into sentences."""
        parts = SWEPruner.SENTENCE_SPLITTER.split(text)
        return [p.strip() for p in parts if p.strip()]

    @staticmethod
    def _fact_importance(sentence: str) -> float:
        """Score a sentence's factual importance (0.0–1.0).

        Boosts for: numbers, dates, proper nouns (capitalised words),
        definition patterns ("is", "means", "refers to").
        """
        score = 0.0
        tokens = sentence.split()
        if not tokens:
            return 0.0

        # Number presence
        if re.search(r"\d+", sentence):
            score += 0.25

        # Date-like patterns
        if re.search(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", sentence):
            score += 0.15

        # Capitalised words (potential proper nouns)
        caps = sum(1 for t in tokens if t[0].isupper())
        ratio_caps = caps / max(len(tokens), 1)
        score += ratio_caps * 0.25

        # Definition patterns
        if re.search(r"\b(is|means|refers to|defined as|equals)\b", sentence, re.I):
            score += 0.2

        # Penalise very short sentences
        if len(tokens) < 4:
            score -= 0.15

        # Reward reasonable length
        if 6 <= len(tokens) <= 25:
            score += 0.1

        return max(0.0, min(score, 1.0))