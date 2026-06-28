"""TZKV runtime-state protection hooks for vLLM."""

from vllm.tzkv.runtime import (
    approve_descriptor,
    commit_block_table,
    enabled,
    verify_block_table,
    verify_descriptor_use,
)

__all__ = [
    "approve_descriptor",
    "commit_block_table",
    "enabled",
    "verify_block_table",
    "verify_descriptor_use",
]
