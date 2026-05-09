# Svalinn — Shield Before the Sun ⚔️

**Context pruning & summarization for LLM pipelines.**

Svalinn is the mythical shield in Norse mythology that stands before the sun to protect the world. Similarly, **SWEPruner** shields your context window from being consumed by unnecessary tokens — keeping only what matters.

## Installation

```bash
pip install svalinn
```

## Quick Start

```python
from svalinn import SWEPruner

pruner = SWEPruner()

# Estimate tokens
count = pruner.token_count("The quick brown fox jumps over the lazy dog.")

# Prune long context to a token budget
short = pruner.prune(long_context, max_tokens=4096)

# Summarize to ~30% of original length
summary = pruner.summarize(document, ratio=0.3)

# Extract key facts
facts = pruner.extract_key_facts(document, max_facts=10)
for f in facts:
    print(f"[{f.importance:.2f}] Line {f.source_line}: {f.content}")
```

## API

| Method | Description |
|--------|-------------|
| `prune(context, max_tokens)` | Reduce context to fit within a token budget |
| `summarize(context, ratio)` | Extractive summarization (ratio 0.0–1.0) |
| `extract_key_facts(context, max_facts)` | Pull out the most important factual sentences |
| `token_count(text)` | Estimate token count using word-ratio heuristic |

## Name

*In Norse mythology, Svalinn is the shield placed before the sun to prevent it from burning the world. It stands as a guardian between overwhelming radiance and what must be preserved.*

## License

MIT