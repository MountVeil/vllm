from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


def enabled() -> bool:
    return os.getenv("TZKV_ENABLE", "0").lower() in {"1", "true", "yes", "on"}


def _strict() -> bool:
    return os.getenv("TZKV_STRICT", "1").lower() in {"1", "true", "yes", "on"}


def _log_path() -> Path:
    return Path(os.getenv("TZKV_LOG", "logs/tzkv_vllm.jsonl"))


def _event(event: str, **fields: Any) -> None:
    if not enabled():
        return
    path = _log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {"ts": time.time(), "event": event, **fields}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def _fail(event: str, reason: str, **fields: Any) -> bool:
    _event(event, reason=reason, **fields)
    if _strict():
        raise RuntimeError(f"TZKV {event}: {reason}")
    return False


def _tensor_bytes(value: torch.Tensor) -> bytes:
    tensor = value.detach()
    if tensor.is_cuda:
        tensor = tensor.cpu()
    tensor = tensor.contiguous()
    arr = tensor.numpy()
    header = json.dumps(
        {"dtype": str(arr.dtype), "shape": list(arr.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return header + b"\0" + arr.tobytes(order="C")


def _hash_numpy(value: np.ndarray) -> bytes:
    arr = np.ascontiguousarray(value)
    header = json.dumps(
        {"dtype": str(arr.dtype), "shape": list(arr.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(header + b"\0" + arr.tobytes(order="C")).digest()


def _hash_any(value: Any) -> bytes:
    if value is None:
        return hashlib.sha256(b"none").digest()
    if isinstance(value, torch.Tensor):
        return hashlib.sha256(b"tensor\0" + _tensor_bytes(value)).digest()
    if isinstance(value, np.ndarray):
        return hashlib.sha256(b"numpy\0" + _hash_numpy(value)).digest()
    if isinstance(value, dict):
        h = hashlib.sha256(b"dict\0")
        for key in sorted(value, key=str):
            h.update(str(key).encode())
            h.update(_hash_any(value[key]))
        return h.digest()
    if isinstance(value, (list, tuple)):
        h = hashlib.sha256(b"seq\0")
        for item in value:
            h.update(_hash_any(item))
        return h.digest()
    return hashlib.sha256(repr(value).encode()).digest()


@dataclass(frozen=True)
class DescriptorApproval:
    slot_mapping_hash: bytes
    descriptor_hash: bytes
    tag: bytes


_runtime_key = os.urandom(32)
_block_table_roots: dict[int, bytes] = {}


def commit_block_table(owner: object, block_table_np: np.ndarray, num_reqs: int) -> None:
    if not enabled():
        return
    effective = np.ascontiguousarray(block_table_np[:num_reqs])
    root = _hash_numpy(effective)
    _block_table_roots[id(owner)] = root
    _event(
        "bt_transition",
        owner=id(owner),
        num_reqs=num_reqs,
        effective_bt_root=root.hex(),
    )


def _current_block_table_root(owner: object, block_table_np: np.ndarray,
                              num_reqs: int) -> bytes:
    return _hash_numpy(np.ascontiguousarray(block_table_np[:num_reqs]))


def verify_block_table(owner: object, block_table_np: np.ndarray,
                       num_reqs: int) -> bool:
    if not enabled():
        return True
    expected = _block_table_roots.get(id(owner))
    current = _current_block_table_root(owner, block_table_np, num_reqs)
    if expected is None:
        _block_table_roots[id(owner)] = current
        return True
    if not hmac.compare_digest(expected, current):
        return _fail(
            "bt_effective_verify_fail",
            "block table changed after commit",
            owner=id(owner),
            expected=expected.hex(),
            current=current.hex(),
        )
    _event("bt_effective_verify", owner=id(owner), effective_bt_root=current.hex())
    return True


def approve_descriptor(slot_mappings: Any, descriptor: Any) -> DescriptorApproval | None:
    if not enabled():
        return None
    slot_hash = _hash_any(slot_mappings)
    desc_hash = _hash_any(descriptor)
    tag = hmac.new(_runtime_key, slot_hash + desc_hash, hashlib.sha256).digest()
    _event(
        "desc_approve",
        slot_mapping_hash=slot_hash.hex(),
        descriptor_hash=desc_hash.hex(),
    )
    return DescriptorApproval(slot_hash, desc_hash, tag)


def verify_descriptor_use(approval: DescriptorApproval | None,
                          slot_mappings: Any,
                          descriptor: Any) -> bool:
    if not enabled() or approval is None:
        return True
    slot_hash = _hash_any(slot_mappings)
    desc_hash = _hash_any(descriptor)
    tag = hmac.new(_runtime_key, slot_hash + desc_hash, hashlib.sha256).digest()
    if not hmac.compare_digest(approval.slot_mapping_hash, slot_hash):
        return _fail("desc_verify_fail", "slot mapping changed before use")
    if not hmac.compare_digest(approval.descriptor_hash, desc_hash):
        return _fail("desc_verify_fail", "descriptor changed before use")
    if not hmac.compare_digest(approval.tag, tag):
        return _fail("desc_verify_fail", "approval tag mismatch")
    _event(
        "desc_verify_use",
        slot_mapping_hash=slot_hash.hex(),
        descriptor_hash=desc_hash.hex(),
    )
    return True
