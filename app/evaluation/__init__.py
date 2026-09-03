"""Offline RAG quality evaluation primitives.

These modules deliberately stay outside the online request path.  They make
release checks reproducible without adding an LLM judge or a new framework to
the production dependency graph.
"""

from app.evaluation.models import EvaluationCase, load_evaluation_cases
from app.evaluation.semantic_support import SemanticSupportCase, load_semantic_support_cases

__all__ = [
    "EvaluationCase",
    "SemanticSupportCase",
    "load_evaluation_cases",
    "load_semantic_support_cases",
]
