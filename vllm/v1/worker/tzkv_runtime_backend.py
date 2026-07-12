from __future__ import annotations

import os
from dataclasses import dataclass
from threading import Lock
from typing import Protocol


class TZKVBackendError(RuntimeError):
    pass


@dataclass(frozen=True)
class BlockTableCommit:
    batch_id: int
    group_id: int
    root: bytes
    num_reqs: int


@dataclass(frozen=True)
class DescriptorApproval:
    batch_id: int
    root: bytes
    record_count: int


class RuntimeBackend(Protocol):
    def commit_block_table(
        self,
        commit: BlockTableCommit,
    ) -> None:
        ...

    def verify_block_table(
        self,
        commit: BlockTableCommit,
    ) -> None:
        ...

    def approve_descriptor(
        self,
        approval: DescriptorApproval,
    ) -> None:
        ...

    def verify_descriptor_use(
        self,
        approval: DescriptorApproval,
    ) -> None:
        ...


class LocalRuntimeBackend:
    """In-process reference backend used before OP-TEE integration."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._bt_roots: dict[tuple[int, int], bytes] = {}
        self._descriptor_roots: dict[int, bytes] = {}

    def commit_block_table(
        self,
        commit: BlockTableCommit,
    ) -> None:
        key = (commit.batch_id, commit.group_id)

        with self._lock:
            self._bt_roots[key] = bytes(commit.root)

    def verify_block_table(
        self,
        commit: BlockTableCommit,
    ) -> None:
        key = (commit.batch_id, commit.group_id)

        with self._lock:
            expected = self._bt_roots.get(key)

        if expected is None:
            raise TZKVBackendError(
                "block-table approval not found"
            )

        if expected != commit.root:
            raise TZKVBackendError(
                "block-table root mismatch"
            )

    def approve_descriptor(
        self,
        approval: DescriptorApproval,
    ) -> None:
        with self._lock:
            self._descriptor_roots[approval.batch_id] = bytes(
                approval.root
            )

    def verify_descriptor_use(
        self,
        approval: DescriptorApproval,
    ) -> None:
        with self._lock:
            expected = self._descriptor_roots.get(
                approval.batch_id
            )

        if expected is None:
            raise TZKVBackendError(
                "descriptor approval not found"
            )

        if expected != approval.root:
            raise TZKVBackendError(
                "descriptor root mismatch"
            )


class OpteeRuntimeBackend:
    """Placeholder until the CA shared library is connected."""

    def __init__(self) -> None:
        raise TZKVBackendError(
            "OP-TEE runtime backend is not connected yet"
        )


_backend: RuntimeBackend | None = None


def get_runtime_backend() -> RuntimeBackend:
    global _backend

    if _backend is not None:
        return _backend

    name = os.getenv(
        "TZKV_RUNTIME_BACKEND",
        "local",
    ).strip().lower()

    if name == "local":
        _backend = LocalRuntimeBackend()
    elif name == "optee":
        _backend = OpteeRuntimeBackend()
    else:
        raise TZKVBackendError(
            f"unsupported TZKV_RUNTIME_BACKEND={name!r}"
        )

    return _backend
