# SPDX-License-Identifier: Apache-2.0
#
# TZKV OP-TEE block-table defense for vLLM.
#
# Security goal:
#   REE may compute and use block tables, but every effective block table
#   consumed by attention must be committed to and verified by OP-TEE.
#
# This first defense module protects BT metadata only. Descriptor and KV payload
# defense should be layered after this.

from __future__ import annotations

import csv
import hashlib
import json
import os
import socket
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None


TZKV_FLAG_NONE = 0x00000000
TZKV_FLAG_STRICT = 0x00000001
TZKV_FLAG_COMMIT = 0x00000002
TZKV_BT_TRANS_APPEND = 1
TZKV_HASH_ZERO = "00" * 32


def _env_bool(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


def _u64_from_text(text: str) -> int:
    h = hashlib.blake2b(text.encode("utf-8", "replace"), digest_size=8).digest()
    return int.from_bytes(h, "little", signed=False)


def _stable_session_id() -> int:
    explicit = os.getenv("TZKV_OPTEE_BT_SESSION_ID")
    if explicit:
        return int(explicit, 0)
    seed = f"{socket.gethostname()}:{os.getpid()}:{time.time_ns()}"
    return _u64_from_text(seed)


@dataclass
class TzkvBtGuardConfig:
    enabled: bool
    strict: bool
    helper: str
    log_path: str
    session_id: int
    block_size: int
    max_entries: int
    reset_on_init: bool
    commit_mode: str
    tamper_after_commit: bool
    tamper_once: bool

    @staticmethod
    def from_env() -> "TzkvBtGuardConfig":
        return TzkvBtGuardConfig(
            enabled=_env_bool("TZKV_OPTEE_BT_ENABLE", "0"),
            strict=_env_bool("TZKV_OPTEE_BT_STRICT", "1"),
            helper=os.getenv(
                "TZKV_OPTEE_BT_HELPER",
                "/home/xx/tee-llm-thor/ca/tzkv_bt_guard_tool",
            ),
            log_path=os.getenv(
                "TZKV_OPTEE_BT_LOG",
                "/home/xx/tee-llm-thor/logs/vllm_bt_defense.jsonl",
            ),
            session_id=_stable_session_id(),
            block_size=int(os.getenv("TZKV_OPTEE_BT_BLOCK_SIZE", "16")),
            max_entries=int(os.getenv("TZKV_OPTEE_BT_MAX_ENTRIES", "256")),
            reset_on_init=_env_bool("TZKV_OPTEE_BT_RESET_ON_INIT", "0"),
            commit_mode=os.getenv("TZKV_OPTEE_BT_COMMIT_MODE", "root").strip().lower(),
            tamper_after_commit=_env_bool("TZKV_OPTEE_BT_TAMPER_AFTER_COMMIT", "0"),
            tamper_once=_env_bool("TZKV_OPTEE_BT_TAMPER_ONCE", "1"),
        )


class TzkvOpteeBtGuard:
    def __init__(self, cfg: Optional[TzkvBtGuardConfig] = None) -> None:
        self.cfg = cfg or TzkvBtGuardConfig.from_env()
        self._lock = threading.Lock()
        self._initialized = False
        self._epoch = 0
        self._root_hex: Optional[str] = None
        self._last_shape: Optional[Tuple[int, ...]] = None
        self._last_req_ids: List[int] = []
        self._tamper_done = False

        if self.cfg.enabled:
            Path(self.cfg.log_path).parent.mkdir(parents=True, exist_ok=True)
            self._log("guard_config", {
                "strict": self.cfg.strict,
                "helper": self.cfg.helper,
                "session_id": self.cfg.session_id,
                "block_size": self.cfg.block_size,
                "max_entries": self.cfg.max_entries,
                "reset_on_init": self.cfg.reset_on_init,
                "commit_mode": self.cfg.commit_mode,
                "tamper_after_commit": self.cfg.tamper_after_commit,
                "tamper_once": self.cfg.tamper_once,
            })

            if not Path(self.cfg.helper).exists():
                self._fail(f"BT helper does not exist: {self.cfg.helper}")

            if self.cfg.reset_on_init:
                self._run_tool_reset()

    def enabled(self) -> bool:
        return self.cfg.enabled

    def commit_and_verify(
        self,
        *,
        runner: Any,
        block_table: Any,
        phase: str,
        layer_id: int = 0,
    ) -> bool:
        if not self.cfg.enabled:
            return True

        with self._lock:
            bt_np = self._to_numpy_int64(block_table)
            shape = tuple(int(x) for x in bt_np.shape)
            req_ids = self._extract_request_ids(runner, expected_rows=shape[0] if shape else 0)
            root_hex = self._compute_bt_root(bt_np, req_ids=req_ids, layer_id=layer_id)

            entries_path = None
            try:
                entries_path, num_entries, num_blocks = self._write_entries_csv(
                    bt_np, req_ids=req_ids, layer_id=layer_id
                )

                if not self._initialized:
                    self._call_init(epoch=0, root_hex=root_hex)
                    self._initialized = True
                    self._epoch = 0
                    self._root_hex = root_hex
                    self._last_shape = shape
                    self._last_req_ids = req_ids
                    self._log("bt_init_ok", {
                        "phase": phase,
                        "epoch": self._epoch,
                        "root": root_hex,
                        "shape": shape,
                        "num_entries": num_entries,
                        "num_blocks": num_blocks,
                    })
                elif root_hex != self._root_hex:
                    old_epoch = self._epoch
                    new_epoch = old_epoch + 1
                    old_root = self._root_hex or TZKV_HASH_ZERO

                    if self.cfg.commit_mode == "semantic":
                        # Full semantic BT transition. This requires KV blocks
                        # to have already been committed to TA.
                        self._call_commit(
                            old_epoch=old_epoch,
                            new_epoch=new_epoch,
                            old_root_hex=old_root,
                            new_root_hex=root_hex,
                            request_id=req_ids[0] if req_ids else 0,
                            sequence_id=0,
                            layer_id=layer_id,
                            token_start=0,
                            token_end=self._estimate_token_end(bt_np),
                            block_size=self._block_size(runner),
                            num_blocks=num_blocks,
                            entries_path=entries_path,
                        )
                        event = "bt_semantic_commit_ok"
                    else:
                        # BT-only root integrity mode. Commit the effective
                        # block-table root into TEE, then verify it immediately
                        # before attention uses it.
                        self._call_init(epoch=new_epoch, root_hex=root_hex)
                        event = "bt_root_commit_ok"

                    self._epoch = new_epoch
                    self._root_hex = root_hex
                    self._last_shape = shape
                    self._last_req_ids = req_ids
                    self._log(event, {
                        "phase": phase,
                        "old_epoch": old_epoch,
                        "new_epoch": new_epoch,
                        "old_root": old_root,
                        "new_root": root_hex,
                        "shape": shape,
                        "num_entries": num_entries,
                        "num_blocks": num_blocks,
                        "commit_mode": self.cfg.commit_mode,
                    })

                # Effective verification is always performed immediately before use.
                effective_root_hex = root_hex
                if self.cfg.tamper_after_commit and (
                    not self.cfg.tamper_once or not self._tamper_done
                ):
                    effective_root_hex = self._tamper_root_hex_for_defense_test(root_hex)
                    self._tamper_done = True
                    self._log("bt_defense_tamper_injected", {
                        "phase": phase,
                        "epoch": self._epoch,
                        "committed_root": root_hex,
                        "effective_root": effective_root_hex,
                        "shape": shape,
                        "num_entries": num_entries,
                        "num_blocks": num_blocks,
                    })

                self._call_effective(epoch=self._epoch, root_hex=effective_root_hex)
                self._log("bt_effective_ok", {
                    "phase": phase,
                    "epoch": self._epoch,
                    "root": effective_root_hex,
                    "committed_root": root_hex,
                    "shape": shape,
                    "num_entries": num_entries,
                    "num_blocks": num_blocks,
                })
                return True

            finally:
                if entries_path:
                    try:
                        os.unlink(entries_path)
                    except OSError:
                        pass

    def _to_numpy_int64(self, tensor: Any):
        if np is None:
            self._fail("numpy is required by TZKV BT guard")

        obj = tensor
        try:
            if hasattr(obj, "detach"):
                obj = obj.detach()
            if hasattr(obj, "cpu"):
                obj = obj.cpu()
            if hasattr(obj, "contiguous"):
                obj = obj.contiguous()
            if hasattr(obj, "numpy"):
                obj = obj.numpy()
        except Exception as exc:
            self._fail(f"failed to move block_table tensor to CPU numpy: {exc}")

        arr = np.asarray(obj)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr.astype(np.int64, copy=False)

    def _extract_request_ids(self, runner: Any, expected_rows: int) -> List[int]:
        candidates: List[Any] = []

        for obj in (
            getattr(runner, "input_batch", None),
            getattr(runner, "scheduler_output", None),
            runner,
        ):
            if obj is None:
                continue
            for name in ("req_ids", "request_ids", "requests", "request_id"):
                if hasattr(obj, name):
                    candidates.append(getattr(obj, name))

        for cand in candidates:
            try:
                if isinstance(cand, dict):
                    vals = list(cand.keys())
                elif isinstance(cand, (list, tuple)):
                    vals = list(cand)
                else:
                    vals = list(cand)
                if vals:
                    ids = [_u64_from_text(str(v)) for v in vals[:expected_rows]]
                    while len(ids) < expected_rows:
                        ids.append(len(ids) + 1)
                    return ids
            except Exception:
                continue

        return [i + 1 for i in range(expected_rows)]

    def _compute_bt_root(self, bt_np: Any, *, req_ids: Sequence[int], layer_id: int) -> str:
        h = hashlib.sha256()
        h.update(b"TZKV.vLLM.block_table.v1\0")
        h.update(int(layer_id).to_bytes(8, "little", signed=False))
        h.update(len(bt_np.shape).to_bytes(8, "little", signed=False))
        for dim in bt_np.shape:
            h.update(int(dim).to_bytes(8, "little", signed=False))
        h.update(len(req_ids).to_bytes(8, "little", signed=False))
        for rid in req_ids:
            h.update(int(rid).to_bytes(8, "little", signed=False))
        h.update(bt_np.tobytes(order="C"))
        return h.hexdigest()

    def _write_entries_csv(
        self,
        bt_np: Any,
        *,
        req_ids: Sequence[int],
        layer_id: int,
    ) -> Tuple[str, int, int]:
        block_size = self.cfg.block_size
        session_id = self.cfg.session_id

        fd, path = tempfile.mkstemp(prefix="tzkv_bt_entries_", suffix=".csv")
        os.close(fd)

        num_entries = 0
        num_blocks = 0

        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)

            rows = int(bt_np.shape[0]) if bt_np.ndim >= 2 else 1
            cols = int(bt_np.shape[1]) if bt_np.ndim >= 2 else int(bt_np.size)

            for row in range(rows):
                request_id = int(req_ids[row]) if row < len(req_ids) else row + 1
                for logical_block_id in range(cols):
                    try:
                        physical_block_id = int(bt_np[row, logical_block_id])
                    except Exception:
                        continue

                    # vLLM block tables may contain padding. Negative values are
                    # invalid physical blocks and are ignored as unused entries.
                    if physical_block_id < 0:
                        continue

                    token_start = logical_block_id * block_size
                    token_end = token_start + block_size

                    w.writerow([
                        request_id,          # request_id
                        0,                   # source_request_id; APPEND has no source request
                        row,                 # sequence_id
                        layer_id,            # layer_id
                        logical_block_id,    # logical_block_id
                        physical_block_id,   # physical_block_id
                        session_id,          # owner_session_id
                        token_start,         # token_start
                        token_end,           # token_end
                        0,                   # prefix_token_start
                        0,                   # prefix_token_end
                        0,                   # flags
                        TZKV_HASH_ZERO,      # prefix_hash
                    ])

                    num_entries += 1
                    num_blocks += 1

                    if num_entries >= self.cfg.max_entries:
                        break
                if num_entries >= self.cfg.max_entries:
                    break

        return path, num_entries, num_blocks

    def _block_size(self, runner: Any) -> int:
        for obj in (runner, getattr(runner, "model_config", None),
                    getattr(runner, "cache_config", None)):
            if obj is None:
                continue
            for name in ("block_size", "kv_cache_block_size"):
                v = getattr(obj, name, None)
                if v:
                    try:
                        return int(v)
                    except Exception:
                        pass
        return int(self.cfg.block_size)

    def _estimate_token_end(self, bt_np: Any) -> int:
        if bt_np.ndim < 2:
            return int(bt_np.size) * self.cfg.block_size
        return int(bt_np.shape[1]) * self.cfg.block_size

    def _tamper_root_hex_for_defense_test(self, root_hex: str) -> str:
        """Return a different 32-byte hex digest to simulate post-commit BT tamper.

        This does not mutate the real block_table tensor. It intentionally asks
        the TEE to verify a different effective root after committing the legal
        root, exercising the same fail-stop path as post-commit runtime BT
        mutation.
        """
        b = bytearray.fromhex(root_hex)
        if not b:
            return root_hex
        b[0] ^= 0x01
        return b.hex()

    def _call_init(self, *, epoch: int, root_hex: str) -> dict:
        return self._run_helper([
            "init",
            str(self.cfg.session_id),
            str(epoch),
            root_hex,
            str(TZKV_FLAG_STRICT if self.cfg.strict else TZKV_FLAG_NONE),
        ])

    def _call_effective(self, *, epoch: int, root_hex: str) -> dict:
        return self._run_helper([
            "effective",
            str(self.cfg.session_id),
            str(epoch),
            root_hex,
            str(TZKV_FLAG_STRICT if self.cfg.strict else TZKV_FLAG_NONE),
        ])

    def _call_commit(
        self,
        *,
        old_epoch: int,
        new_epoch: int,
        old_root_hex: str,
        new_root_hex: str,
        request_id: int,
        sequence_id: int,
        layer_id: int,
        token_start: int,
        token_end: int,
        block_size: int,
        num_blocks: int,
        entries_path: str,
    ) -> dict:
        return self._run_helper([
            "commit",
            str(self.cfg.session_id),
            str(old_epoch),
            str(new_epoch),
            old_root_hex,
            new_root_hex,
            str(request_id),
            str(sequence_id),
            str(layer_id),
            "0",  # kv_epoch, BT-only defense v1
            str(token_start),
            str(token_end),
            str(block_size),
            str(num_blocks),
            str(TZKV_BT_TRANS_APPEND),
            entries_path,
            str(TZKV_FLAG_COMMIT | (TZKV_FLAG_STRICT if self.cfg.strict else 0)),
        ])

    def _run_helper(self, args: Sequence[str]) -> dict:
        cmd = [self.cfg.helper, *args]
        try:
            cp = subprocess.run(
                cmd,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
        except Exception as exc:
            self._fail(f"failed to execute BT helper: {exc}")

        stdout = cp.stdout.strip()
        stderr = cp.stderr.strip()

        try:
            obj = json.loads(stdout.splitlines()[-1]) if stdout else {}
        except Exception:
            obj = {"parse_error": True, "stdout": stdout}

        obj["returncode"] = cp.returncode
        if stderr:
            obj["stderr"] = stderr

        transport_ok = int(obj.get("transport_status", -1)) == 0
        status_ok = int(obj.get("status", -1)) == 0

        if cp.returncode != 0 or not transport_ok or not status_ok:
            self._log("bt_guard_call_fail", {
                "cmd": cmd,
                "stdout": stdout,
                "stderr": stderr,
                "parsed": obj,
            })
            self._fail(f"BT guard helper failed: {obj}")

        return obj

    def _run_tool_reset(self) -> None:
        tool = os.getenv("TZKV_OPTEE_TOOL", "/home/xx/tee-llm-thor/ca/tzkv_tool")
        try:
            cp = subprocess.run(
                [tool, "reset"],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self._log("reset_on_init", {
                "returncode": cp.returncode,
                "stdout": cp.stdout.strip(),
                "stderr": cp.stderr.strip(),
            })
        except Exception as exc:
            self._fail(f"reset_on_init failed: {exc}")

    def _log(self, event: str, obj: dict) -> None:
        if not self.cfg.enabled:
            return
        rec = {
            "ts": time.time(),
            "event": event,
            **obj,
        }
        try:
            with open(self.cfg.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, sort_keys=True) + "\n")
        except Exception:
            pass

    def _fail(self, msg: str) -> None:
        self._log("bt_guard_error", {"message": msg})
        if self.cfg.strict:
            raise RuntimeError(msg)


_GUARD: Optional[TzkvOpteeBtGuard] = None
_GUARD_LOCK = threading.Lock()


def get_tzkv_optee_bt_guard() -> TzkvOpteeBtGuard:
    global _GUARD
    with _GUARD_LOCK:
        if _GUARD is None:
            _GUARD = TzkvOpteeBtGuard()
        return _GUARD
