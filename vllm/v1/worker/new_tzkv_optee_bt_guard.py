from __future__ import annotations

import hashlib
import json
import os
import struct
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _stable_u64(value: Any) -> int:
    if isinstance(value, int):
        return value & ((1 << 64) - 1)
    if isinstance(value, bytes):
        data = value
    else:
        data = str(value).encode("utf-8", "replace")
    return int.from_bytes(hashlib.blake2b(data, digest_size=8).digest(), "little")


def _h32(value: bytes | str) -> bytes:
    if isinstance(value, str):
        value = value.encode("utf-8", "replace")
    return hashlib.sha256(value).digest()


def _flip_root(root_hex: str) -> str:
    b = bytearray.fromhex(root_hex)
    if not b:
        return root_hex
    b[0] ^= 0x01
    return bytes(b).hex()


class TZKVOPEEBTGuard:
    """vLLM-side semantic BT guard for the current tee-llm-thor backend.

    This class deliberately does NOT use bt-init and does NOT use root-only
    block-table commit.  It registers referenced KV blocks first, then submits a
    semantic block-table transition, and finally verifies the committed effective
    root before attention metadata is consumed.

    The current integration is layer-0 / metadata-payload oriented.  It is enough
    to make the production OP-TEE BT protocol run from vLLM without using the old
    debug/root-only path.  Later, the payload passed to the KV manifest should be
    replaced with the real KV block bytes/hash at the KV write-completion hook.
    """

    def __init__(self) -> None:
        self.enabled = _env_bool("TZKV_OPTEE_BT_ENABLE", False)
        self.strict = _env_bool("TZKV_OPTEE_BT_STRICT", True)
        self.reset_on_init = _env_bool("TZKV_OPTEE_BT_RESET_ON_INIT", False)
        self.tamper_after_commit = _env_bool("TZKV_OPTEE_BT_TAMPER_AFTER_COMMIT", False)
        self.tamper_once = _env_bool("TZKV_OPTEE_BT_TAMPER_ONCE", True)
        self._tampered = False

        self.tee_repo = Path(os.getenv("TZKV_OPTEE_BT_REPO", "/home/xx/tee-llm-thor"))
        self.tool = os.getenv("TZKV_OPTEE_BT_TOOL", str(self.tee_repo / "ca" / "tzkv_tool"))
        self.lib = os.getenv("TZKV_OPTEE_BT_LIB", str(self.tee_repo / "ca" / "libtzkv_client.so"))
        self.log_path = Path(os.getenv("TZKV_OPTEE_BT_LOG", str(self.tee_repo / "logs" / "vllm_bt_defense.jsonl")))
        self.tag_dir = Path(os.getenv("TZKV_OPTEE_TAG_DIR", str(self.tee_repo / "results" / "vllm_tags")))
        self.state_dir = Path(os.getenv("TZKV_OPTEE_STATE_DIR", str(self.tee_repo / "results" / "vllm_state")))

        self.block_size = int(os.getenv("TZKV_OPTEE_BT_BLOCK_SIZE", "16"))
        self.max_entries = int(os.getenv("TZKV_OPTEE_BT_MAX_ENTRIES", "256"))
        self.kv_tree_leaves = int(os.getenv("TZKV_OPTEE_KV_TREE_LEAVES", "8192"))
        self.bt_tree_leaves = int(os.getenv("TZKV_OPTEE_BT_TREE_LEAVES", "8192"))

        self.session_id = int(os.getenv("TZKV_OPTEE_SESSION_ID", "0") or "0")
        if self.session_id == 0:
            self.session_id = int.from_bytes(os.urandom(8), "little") & ((1 << 63) - 1)

        self.app_hash = _h32(os.getenv("TZKV_APP_ID", "vllm-tzkv-app"))
        self.model_hash = _h32(os.getenv("TZKV_MODEL_ID", "vllm-model"))
        self.tokenizer_hash = _h32(os.getenv("TZKV_TOKENIZER_ID", "vllm-tokenizer"))
        self.prompt_hash = _h32(os.getenv("TZKV_PROMPT_ID", "vllm-runtime-prompt"))

        self._initialized = False
        self._client = None
        self._policy = None
        self._kv_tree = None
        self._bt_tree = None
        self._kv_guard = None
        self._bt_guard = None

        self._kv_epoch = 0
        self._next_kv_leaf = 0
        self._kv_committed: dict[tuple[int, int, int, int, int], dict] = {}
        self._bt_records: dict[tuple[int, int, int], dict] = {}

        self._log("guard_config", {
            "enabled": self.enabled,
            "strict": self.strict,
            "session_id": self.session_id,
            "tee_repo": str(self.tee_repo),
            "tool": self.tool,
            "lib": self.lib,
            "block_size": self.block_size,
            "max_entries": self.max_entries,
            "tamper_after_commit": self.tamper_after_commit,
        })

    def commit_and_verify(self, *, runner: Any, block_table: Any, phase: str, layer_id: int = 0) -> bool:
        if not self.enabled:
            return True
        try:
            self._ensure_initialized()
            rows = self._block_table_rows(block_table)
            req_ids = self._request_ids(runner, len(rows))
            if not rows:
                self._log("bt_skip_empty", {"phase": phase, "layer_id": layer_id})
                return True

            all_ok = True
            for row_idx, row in enumerate(rows):
                raw_req_id = req_ids[row_idx] if row_idx < len(req_ids) else f"row-{row_idx}"
                request_id = _stable_u64(raw_req_id)
                block_ids = self._valid_block_ids(row)
                if not block_ids:
                    self._log("bt_skip_row_empty", {
                        "phase": phase,
                        "row": row_idx,
                        "request_id": request_id,
                        "raw_request_id": str(raw_req_id),
                    })
                    continue
                ok = self._commit_verify_one_request(
                    request_id=request_id,
                    raw_request_id=str(raw_req_id),
                    sequence_id=0,
                    layer_id=layer_id,
                    block_ids=block_ids,
                    phase=phase,
                    row_idx=row_idx,
                )
                all_ok = bool(all_ok and ok)
            return all_ok
        except Exception as exc:
            self._log("bt_guard_error", {"phase": phase, "layer_id": layer_id, "message": repr(exc)})
            if self.strict:
                raise
            return False

    def _commit_verify_one_request(
        self,
        *,
        request_id: int,
        raw_request_id: str,
        sequence_id: int,
        layer_id: int,
        block_ids: list[int],
        phase: str,
        row_idx: int,
    ) -> bool:
        assert self._client is not None and self._kv_guard is not None and self._bt_guard is not None
        block_ids = block_ids[: self.max_entries]
        token_start = 0
        token_end = len(block_ids) * self.block_size

        for logical_idx, physical_block_id in enumerate(block_ids):
            self._ensure_kv_committed(
                request_id=request_id,
                raw_request_id=raw_request_id,
                layer_id=layer_id,
                logical_block_id=logical_idx,
                physical_block_id=physical_block_id,
                token_start=logical_idx * self.block_size,
                token_end=(logical_idx + 1) * self.block_size,
                phase=phase,
            )

        state = self._client.get_session_state(self.session_id)
        self._check_resp("get-session-state", state, {"phase": phase, "request_id": request_id})
        kv_epoch = int(state.get("kv_epoch", self._kv_epoch))

        key = (request_id, sequence_id, layer_id)
        prev = self._bt_records.get(key)
        prev_blocks = prev.get("block_ids") if prev else None

        if prev and prev_blocks == block_ids:
            committed_epoch = int(prev["bt_epoch"])
            committed_root = str(prev["bt_root"])
            self._log("bt_commit_skip_unchanged", {
                "phase": phase,
                "row": row_idx,
                "request_id": request_id,
                "layer_id": layer_id,
                "bt_epoch": committed_epoch,
                "bt_root": committed_root,
                "num_blocks": len(block_ids),
            })
        else:
            old_epoch = int(prev["bt_epoch"]) if prev else 0
            old_root = str(prev["bt_root"]) if prev else self._bt_tree.get_root().hex()
            transition = self._bt_guard.build_semantic_transition(
                session_id=self.session_id,
                request_id=request_id,
                sequence_id=sequence_id,
                block_ids=block_ids,
                token_start=token_start,
                token_end=token_end,
                block_size=self.block_size,
                kv_epoch=kv_epoch,
                layer_id=layer_id,
                app_hash=self.app_hash,
                old_bt_epoch=old_epoch,
                old_bt_root=old_root,
                owner_session_id=self.session_id,
                flags=0,
                transition_type=1,
            )
            resp = self._client.bt_commit_transition(transition)
            self._check_resp("bt-commit", resp, {
                "phase": phase,
                "request_id": request_id,
                "old_bt_epoch": old_epoch,
                "new_bt_epoch": transition.get("new_bt_epoch"),
                "old_bt_root": old_root,
                "new_bt_root": transition.get("new_bt_root"),
                "num_blocks": len(block_ids),
            })
            committed_epoch = int(resp.get("committed_bt_epoch", transition["new_bt_epoch"]))
            committed_root = str(resp.get("committed_bt_root", transition["new_bt_root"]))
            self._bt_records[key] = {
                "bt_epoch": committed_epoch,
                "bt_root": committed_root,
                "block_ids": list(block_ids),
                "kv_epoch": kv_epoch,
            }
            self._log("bt_commit_transition_ok", {
                "phase": phase,
                "row": row_idx,
                "request_id": request_id,
                "raw_request_id": raw_request_id,
                "layer_id": layer_id,
                "old_bt_epoch": old_epoch,
                "new_bt_epoch": committed_epoch,
                "committed_bt_root": committed_root,
                "num_blocks": len(block_ids),
            })

        effective_root = committed_root
        if self.tamper_after_commit and (not self._tampered or not self.tamper_once):
            effective_root = _flip_root(effective_root)
            self._tampered = True
            self._log("bt_defense_tamper_injected", {
                "phase": phase,
                "request_id": request_id,
                "layer_id": layer_id,
                "committed_bt_root": committed_root,
                "effective_bt_root": effective_root,
            })

        verify = self._client.bt_verify_effective({
            "session_id": self.session_id,
            "request_id": request_id,
            "sequence_id": sequence_id,
            "layer_id": layer_id,
            "bt_epoch": committed_epoch,
            "effective_bt_root": effective_root,
        })
        self._check_resp("bt-effective", verify, {
            "phase": phase,
            "request_id": request_id,
            "layer_id": layer_id,
            "bt_epoch": committed_epoch,
            "effective_bt_root": effective_root,
        })
        self._log("bt_effective_ok", {
            "phase": phase,
            "row": row_idx,
            "request_id": request_id,
            "raw_request_id": raw_request_id,
            "layer_id": layer_id,
            "bt_epoch": committed_epoch,
            "effective_bt_root": effective_root,
            "num_blocks": len(block_ids),
        })
        return True

    def _ensure_kv_committed(
        self,
        *,
        request_id: int,
        raw_request_id: str,
        layer_id: int,
        logical_block_id: int,
        physical_block_id: int,
        token_start: int,
        token_end: int,
        phase: str,
    ) -> None:
        assert self._kv_guard is not None
        key = (request_id, layer_id, physical_block_id, token_start, token_end)
        if key in self._kv_committed:
            return
        if self._next_kv_leaf >= self._kv_tree.num_leaves:
            self._fail(f"KV Merkle tree exhausted: {self._next_kv_leaf}/{self._kv_tree.num_leaves}")
        self._kv_epoch += 1
        leaf_index = self._next_kv_leaf
        self._next_kv_leaf += 1

        payload = self._metadata_payload(
            request_id=request_id,
            raw_request_id=raw_request_id,
            layer_id=layer_id,
            logical_block_id=logical_block_id,
            physical_block_id=physical_block_id,
            token_start=token_start,
            token_end=token_end,
        )
        manifest = self._kv_guard.build_manifest(
            session_id=self.session_id,
            request_id=request_id,
            layer_id=layer_id,
            block_id=physical_block_id,
            leaf_index=leaf_index,
            token_start=token_start,
            token_end=token_end,
            kv_epoch=self._kv_epoch,
            payload=payload,
            app_hash=self.app_hash,
            model_hash=self.model_hash,
            tokenizer_hash=self.tokenizer_hash,
            prompt_hash=self.prompt_hash,
            dtype=0,
            element_size=0,
        )
        seal = self._kv_guard.seal_block(manifest)
        commit = self._kv_guard.commit_block(manifest, seal)
        self._check_resp("kv-commit", commit, {
            "phase": phase,
            "request_id": request_id,
            "layer_id": layer_id,
            "physical_block_id": physical_block_id,
            "token_start": token_start,
            "token_end": token_end,
            "kv_epoch": self._kv_epoch,
        })
        self._kv_committed[key] = {"manifest": manifest, "seal": seal, "commit": commit}
        self._log("kv_commit_ok", {
            "phase": phase,
            "request_id": request_id,
            "raw_request_id": raw_request_id,
            "layer_id": layer_id,
            "logical_block_id": logical_block_id,
            "physical_block_id": physical_block_id,
            "token_start": token_start,
            "token_end": token_end,
            "kv_epoch": self._kv_epoch,
            "leaf_index": leaf_index,
        })

    def _metadata_payload(self, **fields: Any) -> bytes:
        body = b"TZKV.vllm.kv.metadata.v1\0"
        for name in (
            "request_id", "layer_id", "logical_block_id", "physical_block_id",
            "token_start", "token_end",
        ):
            body += struct.pack("<Q", int(fields[name]) & ((1 << 64) - 1))
        raw = str(fields.get("raw_request_id", "")).encode("utf-8", "replace")
        body += hashlib.sha256(raw).digest()
        return body

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        py_dir = self.tee_repo / "python"
        if str(py_dir) not in sys.path:
            sys.path.insert(0, str(py_dir))

        from tzkv_block_table_guard import TZKVBlockTableGuard
        from tzkv_kv_guard import TZKVKVGuard
        from tzkv_merkle import MerkleTree
        from tzkv_optee_client import TZKVClient
        from tzkv_policy import TZKVPolicy

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.tag_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)

        self._client = TZKVClient(tool=self.tool, lib=self.lib, local_fallback=False)
        if self.reset_on_init:
            self._client.reset()
        self._kv_tree = MerkleTree(self.kv_tree_leaves)
        self._bt_tree = MerkleTree(self.bt_tree_leaves)
        self._policy = TZKVPolicy(
            enable=True,
            strict=True,
            abort_on_fail=True,
            log=self.log_path,
            tag_dir=self.tag_dir,
            state_dir=self.state_dir,
            optee_tool=self.tool,
        )
        self._kv_guard = TZKVKVGuard(self._client, self._policy, self._kv_tree)
        self._bt_guard = TZKVBlockTableGuard(self._client, self._policy, self._bt_tree)

        open_req = {
            "session_id": self.session_id,
            "flags": 0,
            "initial_kv_epoch": 0,
            "initial_bt_epoch": 0,
            "initial_desc_epoch": 0,
            "app_hash": self.app_hash.hex(),
            "model_hash": self.model_hash.hex(),
            "tokenizer_hash": self.tokenizer_hash.hex(),
            "prompt_hash": self.prompt_hash.hex(),
            "initial_kv_root": self._kv_tree.get_root().hex(),
            "initial_bt_root": self._bt_tree.get_root().hex(),
            "nonce": hashlib.blake2b(str(self.session_id).encode(), digest_size=16).hexdigest(),
        }
        resp = self._client.open_session(open_req)
        self._check_resp("open-session", resp, {"session_id": self.session_id})
        self._initialized = True
        self._log("bt_session_open_ok", {
            "session_id": self.session_id,
            "initial_kv_root": open_req["initial_kv_root"],
            "initial_bt_root": open_req["initial_bt_root"],
            "resp": resp,
        })

    def _request_ids(self, runner: Any, n: int) -> list[Any]:
        batch = getattr(runner, "input_batch", None)
        req_ids = getattr(batch, "req_ids", None) if batch is not None else None
        if req_ids is None:
            return [f"req-{i}" for i in range(n)]
        try:
            out = list(req_ids)
        except TypeError:
            out = [req_ids]
        if len(out) < n:
            out.extend(f"req-{i}" for i in range(len(out), n))
        return out

    def _block_table_rows(self, block_table: Any) -> list[list[int]]:
        obj = block_table
        if hasattr(obj, "detach"):
            obj = obj.detach()
        if hasattr(obj, "cpu"):
            obj = obj.cpu()
        if hasattr(obj, "numpy"):
            obj = obj.numpy()
        if hasattr(obj, "tolist"):
            obj = obj.tolist()
        if obj is None:
            return []
        if isinstance(obj, tuple):
            obj = list(obj)
        if not isinstance(obj, list):
            return [[int(obj)]]
        if not obj:
            return []
        if obj and not isinstance(obj[0], (list, tuple)):
            return [[int(x) for x in obj]]
        return [[int(x) for x in row] for row in obj]

    def _valid_block_ids(self, row: Sequence[int]) -> list[int]:
        out = []
        for x in row:
            v = int(x)
            if v >= 0:
                out.append(v)
        return out

    def _check_resp(self, command: str, resp: dict, ctx: dict | None = None) -> None:
        status = resp.get("status") if isinstance(resp, dict) else None
        if status == 0:
            self._log("bt_guard_call_ok", {"command": command, "resp": resp, **(ctx or {})})
            return
        self._log("bt_guard_call_fail", {"command": command, "resp": resp, **(ctx or {})})
        self._fail(f"TZKV {command} failed: {resp}")

    def _fail(self, msg: str) -> None:
        if self.strict:
            raise RuntimeError(msg)

    def _log(self, event: str, fields: dict | None = None) -> None:
        row = {"ts": time.time(), "event": event, **(fields or {})}
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, sort_keys=True, default=str) + "\n")
        except Exception:
            pass


_GUARD: TZKVOPEEBTGuard | None = None


def get_tzkv_optee_bt_guard() -> TZKVOPEEBTGuard:
    global _GUARD
    if _GUARD is None:
        _GUARD = TZKVOPEEBTGuard()
    return _GUARD
