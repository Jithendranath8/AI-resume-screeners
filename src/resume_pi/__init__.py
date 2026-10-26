"""resume_pi — a minimal, deterministic reproduction harness for indirect
prompt-injection attacks and defenses in AI resume screening.

Reproduction target (M2 anchor paper):
    Jay Barach (2026). "ResumeShield: Channel Separation and an Open Benchmark
    for Indirect Prompt Injection in AI Resume Screening." arXiv:2609.20188.

Everything here is Python standard library only and CPU-only, matching the
original paper's deterministic *simulated* screener. No GPU, no API key, no
model download, no training.
"""
__all__ = [
    "document", "payloads", "concealment", "screener", "defense", "detect", "metrics",
]
