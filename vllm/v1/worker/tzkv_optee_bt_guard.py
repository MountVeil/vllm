from __future__ import annotations

import hashlib
import json
import os
import struct
import sys
import time
from pathlib import Path
from typing import Any, Sequence


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
    if b:
        b[0] ^= 0x01
    return bytes(b).hex()


TZKV_BT_APPEND = 1
TZKV_BT_REUSE = 2
TZKV_BT_FREE = 3
TZKV_BT_PREFIX_REUSE = 4


class TZKVOPEEBTGuard:
    """Semantic vLLM -> tee-llm-thor BT guard.

    This file intentionally does NOT use:

    Production flow:
      open-session
        -> commit referenced KV metadata blocks
        -> build semantic BT transition
        -> bt-commit-transition
        -> bt-effective before attention consumes metadata

    This is a vLLM-side adapter for the current tee-llm-thor semantics.
    It does not modify tee-llm-thor and does not touch attack scripts.
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
        self.max_entries = int(os.getenv("TZKV_OPTEE_BT_MAX_ENTRIES", "128"))
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

        # KV metadata already committed to TEE.
        # key = (request_id, sequence_id, layer_id, physical_block_id, token_start, token_end)
        self._kv_committed: dict[tuple[int, int, int, int, int, int], dict] = {}

        # Per request/sequence/layer committed BT record.
        # key = (request_id, sequence_id, layer_id)
        self._bt_records: dict[tuple[int, int, int], dict] = {}

        # vLLM request IDs can be reused after abort/resubmit. Keep a monotonically
        # increasing adapter-side sequence namespace for each raw request ID.
        self._request_seq_counter: dict[str, int] = {}
        self._active_sequences: dict[str, int] = {}

        # Adapter-side shadow of physical block ownership. TEE remains authoritative;
        # this catches obvious owner mismatch before invoking OP-TEE and lets us
        # choose APPEND vs REUSE for a newly observed request/block relation.
        # key = (layer_id, physical_block_id)
        self._physical_block_state: dict[tuple[int, int], dict] = {}

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
        """Entry point used by gpu_model_runner.py.

        Existing call site can stay as:
            get_tzkv_optee_bt_guard().commit_and_verify(...)
        """
        if not self.enabled:
            return True

        try:
            self._ensure_initialized()

            rows = self._block_table_rows(block_table)
            req_ids = self._request_ids(runner, len(rows))

            if not rows:
                self._log("bt_skip_empty", {"phase": phase, "layer_id": layer_id})
                return True

            ok_all = True
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

                raw_req_id_str = str(raw_req_id)
                sequence_id = self._get_or_create_sequence_id(raw_req_id_str)

                ok = self._commit_verify_one_request(
                    request_id=request_id,
                    raw_request_id=raw_req_id_str,
                    sequence_id=sequence_id,
                    layer_id=layer_id,
                    block_ids=block_ids,
                    phase=phase,
                    row_idx=row_idx,
                )
                ok_all = ok_all and ok

            return ok_all

        except Exception as exc:
            self._log("bt_guard_error", {
                "phase": phase,
                "layer_id": layer_id,
                "message": repr(exc),
            })
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
        assert self._client is not None
        assert self._kv_guard is not None
        assert self._bt_guard is not None
        assert self._bt_tree is not None

        # vLLM real block tables may use 0 as padding. In our current
        # integration, valid physical block ids start from 1; <=0 is padding.
        raw_block_ids = list(block_ids)
        block_ids = [int(x) for x in raw_block_ids if int(x) > 0]
        block_ids = block_ids[: self.max_entries]

        if not block_ids:
            self._log("bt_no_valid_blocks_skip", {
                "phase": phase,
                "layer_id": layer_id,
                "row": row_idx,
                "raw_request_id": raw_request_id,
                "request_id": request_id,
                "raw_block_ids": raw_block_ids,
            })
            return True

        self._log("bt_block_ids_filtered", {
            "phase": phase,
            "layer_id": layer_id,
            "row": row_idx,
            "raw_request_id": raw_request_id,
            "request_id": request_id,
            "raw_block_ids": raw_block_ids,
            "block_ids": block_ids,
        })

        token_start = 0
        token_end = len(block_ids) * self.block_size

        # 1. First commit referenced KV metadata blocks.
        for logical_idx, physical_block_id in enumerate(block_ids):
            self._ensure_kv_committed(
                request_id=request_id,
                raw_request_id=raw_request_id,
                sequence_id=sequence_id,
                layer_id=layer_id,
                logical_block_id=logical_idx,
                physical_block_id=physical_block_id,
                token_start=logical_idx * self.block_size,
                token_end=(logical_idx + 1) * self.block_size,
                phase=phase,
            )

        # 2. Build semantic BT transition.
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
                "raw_request_id": raw_request_id,
                "layer_id": layer_id,
                "bt_epoch": committed_epoch,
                "bt_root": committed_root,
                "num_blocks": len(block_ids),
            })
        else:
            old_epoch = int(prev["bt_epoch"]) if prev else 0
            # tee-llm-thor's production BT state starts from the zero root.
            # The MerkleTree empty root is used by the Python helper for local
            # construction, but the TA committed BT root before the first
            # transition is 00..00.
            old_root = str(prev["bt_root"]) if prev else ("00" * 32)
            transition_type = self._choose_bt_transition_type(
                request_id=request_id,
                raw_request_id=raw_request_id,
                sequence_id=sequence_id,
                layer_id=layer_id,
                prev_blocks=prev_blocks,
                block_ids=block_ids,
                phase=phase,
            )

            transition = self._bt_guard.build_semantic_transition(
                session_id=self.session_id,
                request_id=request_id,
                sequence_id=sequence_id,
                block_ids=block_ids,
                token_start=token_start,
                token_end=token_end,
                block_size=self.block_size,
                kv_epoch=self._kv_epoch,
                layer_id=layer_id,
                app_hash=self.app_hash,
                old_bt_epoch=old_epoch,
                old_bt_root=old_root,
                owner_session_id=self.session_id,
                flags=0,
                transition_type=transition_type,
            )

            self._log("bt_commit_transition_request", {
                "phase": phase,
                "row": row_idx,
                "request_id": request_id,
                "raw_request_id": raw_request_id,
                "sequence_id": sequence_id,
                "layer_id": layer_id,
                "block_ids": block_ids,
                "transition": transition,
            })

            resp = self._client.bt_commit_transition(transition)
            self._check_resp("bt-commit-transition", resp, {
                "phase": phase,
                "row": row_idx,
                "request_id": request_id,
                "raw_request_id": raw_request_id,
                "sequence_id": sequence_id,
                "layer_id": layer_id,
                "old_bt_epoch": old_epoch,
                "new_bt_epoch": transition.get("new_bt_epoch"),
                "old_bt_root": old_root,
                "new_bt_root": transition.get("new_bt_root"),
                "transition_type": transition_type,
                "num_blocks": len(block_ids),
                "block_ids": block_ids,
                "transition": transition,
            })

            committed_epoch = int(resp.get("committed_bt_epoch", transition["new_bt_epoch"]))
            committed_root = str(resp.get("committed_bt_root", transition["new_bt_root"]))

            self._bt_records[key] = {
                "bt_epoch": committed_epoch,
                "bt_root": committed_root,
                "block_ids": list(block_ids),
                "kv_epoch": self._kv_epoch,
                "raw_request_id": raw_request_id,
            }

            for physical_block_id in block_ids:
                self._physical_block_state[(layer_id, physical_block_id)] = {
                    "state": "active",
                    "request_id": request_id,
                    "sequence_id": sequence_id,
                    "raw_request_id": raw_request_id,
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
                "transition_type": transition_type,
                "num_blocks": len(block_ids),
            })

        # 3. Verify effective root immediately before attention consumes metadata.
        effective_root = committed_root

        if self.tamper_after_commit and (not self._tampered or not self.tamper_once):
            effective_root = _flip_root(effective_root)
            self._tampered = True
            self._log("bt_defense_tamper_injected", {
                "phase": phase,
                "request_id": request_id,
                "raw_request_id": raw_request_id,
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
            "row": row_idx,
            "request_id": request_id,
            "raw_request_id": raw_request_id,
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

    def _get_or_create_sequence_id(self, raw_request_id: str) -> int:
        seq = self._active_sequences.get(raw_request_id)
        if seq is not None:
            return seq

        # Keep the first incarnation of a vLLM request at sequence_id=0.
        # TEE-side record arrays commonly treat sequence_id as a zero-based slot.
        seq = self._request_seq_counter.get(raw_request_id, -1) + 1
        self._request_seq_counter[raw_request_id] = seq
        self._active_sequences[raw_request_id] = seq
        self._log("request_sequence_open", {
            "raw_request_id": raw_request_id,
            "sequence_id": seq,
        })
        return seq

    def _drop_active_sequence(self, raw_request_id: str) -> None:
        self._active_sequences.pop(raw_request_id, None)

    def _choose_bt_transition_type(
        self,
        *,
        request_id: int,
        raw_request_id: str,
        sequence_id: int,
        layer_id: int,
        prev_blocks: list[int] | None,
        block_ids: list[int],
        phase: str,
    ) -> int:
        # Existing record for the same request/sequence/layer means the request's
        # block table is being extended or refreshed. That is still APPEND under
        # tee-llm-thor semantics, but we still reject obvious cross-owner reuse.
        saw_released = False

        for physical_block_id in block_ids:
            state = self._physical_block_state.get((layer_id, physical_block_id))
            if state is None:
                continue

            if state.get("state") == "released":
                saw_released = True
                continue

            if state.get("state") == "active":
                owner = state.get("request_id")
                owner_seq = state.get("sequence_id")
                if owner != request_id or owner_seq != sequence_id:
                    self._fail(
                        f"physical block {physical_block_id} is still active for "
                        f"request={owner}, sequence={owner_seq}, layer={layer_id}; "
                        f"new request={request_id}, sequence={sequence_id}"
                    )

        if prev_blocks is not None:
            transition_type = TZKV_BT_APPEND
        else:
            transition_type = TZKV_BT_REUSE if saw_released else TZKV_BT_APPEND

        self._log("bt_transition_type_selected", {
            "phase": phase,
            "request_id": request_id,
            "raw_request_id": raw_request_id,
            "sequence_id": sequence_id,
            "layer_id": layer_id,
            "transition_type": transition_type,
            "saw_released": saw_released,
            "has_previous_record": prev_blocks is not None,
            "num_blocks": len(block_ids),
        })
        return transition_type

    def finish_request(self, raw_request_id: str, *, phase: str = "request_finished") -> bool:
        """Map vLLM request finish to tee-llm-thor BT_FREE + KV revoke.

        This is deliberately not a TEE request-close command. The namespace is
        still (session_id, request_id, sequence_id, layer_id); this method only
        commits lifecycle transitions using existing TEE commands.
        """
        if not self.enabled:
            return True

        try:
            if not self._initialized:
                self._log("request_finish_skip_uninitialized", {
                    "phase": phase,
                    "raw_request_id": raw_request_id,
                })
                return True

            assert self._client is not None
            assert self._bt_guard is not None

            request_id = _stable_u64(raw_request_id)
            sequence_id = self._active_sequences.get(raw_request_id)
            if sequence_id is None:
                self._log("request_finish_skip_no_active_sequence", {
                    "phase": phase,
                    "request_id": request_id,
                    "raw_request_id": raw_request_id,
                })
                return True

            freed_bt_records = 0
            revoked_kv_records = 0

            for key, rec in list(self._bt_records.items()):
                rec_request_id, rec_sequence_id, layer_id = key
                if rec_request_id != request_id or rec_sequence_id != sequence_id:
                    continue

                block_ids = list(rec.get("block_ids", []))
                if not block_ids:
                    del self._bt_records[key]
                    continue

                old_epoch = int(rec["bt_epoch"])
                old_root = str(rec["bt_root"])
                token_start = 0
                token_end = len(block_ids) * self.block_size

                transition = self._bt_guard.build_semantic_transition(
                    session_id=self.session_id,
                    request_id=request_id,
                    sequence_id=sequence_id,
                    block_ids=block_ids,
                    token_start=token_start,
                    token_end=token_end,
                    block_size=self.block_size,
                    kv_epoch=int(rec.get("kv_epoch", self._kv_epoch)),
                    layer_id=layer_id,
                    app_hash=self.app_hash,
                    old_bt_epoch=old_epoch,
                    old_bt_root=old_root,
                    owner_session_id=self.session_id,
                    flags=0,
                    transition_type=TZKV_BT_FREE,
                )

                resp = self._client.bt_commit_transition(transition)
                self._check_resp("bt-free-transition", resp, {
                    "phase": phase,
                    "request_id": request_id,
                    "raw_request_id": raw_request_id,
                    "sequence_id": sequence_id,
                    "layer_id": layer_id,
                    "old_bt_epoch": old_epoch,
                    "new_bt_epoch": transition.get("new_bt_epoch"),
                    "old_bt_root": old_root,
                    "new_bt_root": transition.get("new_bt_root"),
                    "num_blocks": len(block_ids),
                })

                for physical_block_id in block_ids:
                    self._physical_block_state[(layer_id, physical_block_id)] = {
                        "state": "released",
                        "last_request_id": request_id,
                        "last_sequence_id": sequence_id,
                        "raw_request_id": raw_request_id,
                    }

                del self._bt_records[key]
                freed_bt_records += 1

            for key, rec in list(self._kv_committed.items()):
                (kv_request_id, kv_sequence_id, layer_id,
                 physical_block_id, token_start, token_end) = key
                if kv_request_id != request_id or kv_sequence_id != sequence_id:
                    continue

                manifest = rec.get("manifest")
                revoke_resp = self._revoke_kv_manifest(manifest)
                self._check_resp("kv-revoke", revoke_resp, {
                    "phase": phase,
                    "request_id": request_id,
                    "raw_request_id": raw_request_id,
                    "sequence_id": sequence_id,
                    "layer_id": layer_id,
                    "physical_block_id": physical_block_id,
                    "token_start": token_start,
                    "token_end": token_end,
                })

                # kv-revoke advances the TA session-wide kv_epoch. Keep the
                # Python-side allocator in sync; otherwise the next request may
                # reuse an already-consumed epoch and TA returns EPOCH_REPLAY.
                if isinstance(revoke_resp, dict) and "verified_kv_epoch" in revoke_resp:
                    try:
                        self._kv_epoch = max(self._kv_epoch, int(revoke_resp["verified_kv_epoch"]))
                    except Exception:
                        pass

                del self._kv_committed[key]
                revoked_kv_records += 1

            self._drop_active_sequence(raw_request_id)
            self._log("request_finish_cleanup_ok", {
                "phase": phase,
                "request_id": request_id,
                "raw_request_id": raw_request_id,
                "sequence_id": sequence_id,
                "freed_bt_records": freed_bt_records,
                "revoked_kv_records": revoked_kv_records,
            })
            return True

        except Exception as exc:
            self._log("request_finish_cleanup_error", {
                "phase": phase,
                "raw_request_id": raw_request_id,
                "message": repr(exc),
            })
            if self.strict:
                raise
            return False

    def _revoke_kv_manifest(self, manifest: dict) -> dict:
        if self._kv_guard is not None and hasattr(self._kv_guard, "revoke_block"):
            return self._kv_guard.revoke_block(manifest)
        if self._client is not None and hasattr(self._client, "kv_revoke"):
            return self._client.kv_revoke(manifest)
        if self._client is not None and hasattr(self._client, "revoke_block"):
            return self._client.revoke_block(manifest)
        self._fail("TZKV KV revoke wrapper is missing: expected kv_guard.revoke_block(), client.kv_revoke(), or client.revoke_block()")
        return {"status": 1, "error": "missing kv revoke wrapper"}

    def _normalize_bt_transition_updates(self, transition: dict, *, phase: str) -> None:
        """Ensure the OP-TEE client sees actual BT updates.

        tzkv_block_table_guard.build_semantic_transition() may return the full
        semantic entries under "entries" while leaving "updates" empty. The TEE
        commit path consumes updates, so an APPEND/FREE/REUSE transition with
        entries but no updates is rejected before the BT epoch/root advances.
        """
        if not isinstance(transition, dict):
            return

        entries = transition.get("entries") or []
        updates = transition.get("updates") or []

        if entries and not updates:
            transition["updates"] = [dict(e) for e in entries]
            transition["num_updates"] = len(transition["updates"])
            self._log("bt_transition_updates_filled_from_entries", {
                "phase": phase,
                "num_entries": len(entries),
                "num_updates": len(transition["updates"]),
                "transition_type": transition.get("transition_type"),
                "request_id": transition.get("request_id"),
                "sequence_id": transition.get("sequence_id"),
                "layer_id": transition.get("layer_id"),
            })
        elif updates:
            transition["num_updates"] = len(updates)

    def _ensure_kv_committed(
        self,
        *,
        request_id: int,
        raw_request_id: str,
        sequence_id: int,
        layer_id: int,
        logical_block_id: int,
        physical_block_id: int,
        token_start: int,
        token_end: int,
        phase: str,
    ) -> None:
        assert self._kv_guard is not None
        assert self._kv_tree is not None

        key = (request_id, sequence_id, layer_id, physical_block_id, token_start, token_end)
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
            "raw_request_id": raw_request_id,
            "sequence_id": sequence_id,
            "layer_id": layer_id,
            "physical_block_id": physical_block_id,
            "token_start": token_start,
            "token_end": token_end,
            "kv_epoch": self._kv_epoch,
        })

        self._kv_committed[key] = {
            "manifest": manifest,
            "seal": seal,
            "commit": commit,
        }

        self._log("kv_commit_ok", {
            "phase": phase,
            "request_id": request_id,
            "raw_request_id": raw_request_id,
            "sequence_id": sequence_id,
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
            "request_id",
            "layer_id",
            "logical_block_id",
            "physical_block_id",
            "token_start",
            "token_end",
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
            strict=self.strict,
            abort_on_fail=self.strict,
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
            # tee-llm-thor BT transition state starts from the production zero root.
            # Keep this consistent with the first transition's old_bt_root="00"*32.
            "initial_bt_root": "00" * 32,
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
        out: list[int] = []
        seen: set[int] = set()
        for x in row:
            v = int(x)
            if v < 0:
                continue
            if v in seen:
                # tee-llm-thor rejects duplicate physical block ids in one BT transition.
                continue
            seen.add(v)
            out.append(v)
        return out

    def _check_resp(self, command: str, resp: dict, ctx: dict | None = None) -> None:
        status = resp.get("status") if isinstance(resp, dict) else None
        if status == 0:
            self._log("bt_guard_call_ok", {
                "command": command,
                "resp": resp,
                **(ctx or {}),
            })
            return

        self._log("bt_guard_call_fail", {
            "command": command,
            "resp": resp,
            **(ctx or {}),
        })
        self._fail(f"TZKV {command} failed: {resp}")

    def _fail(self, msg: str) -> None:
        if self.strict:
            raise RuntimeError(msg)

    def _log(self, event: str, fields: dict | None = None) -> None:
        row = {
            "ts": time.time(),
            "event": event,
            **(fields or {}),
        }

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
