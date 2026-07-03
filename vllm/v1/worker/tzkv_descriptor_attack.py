# SPDX-License-Identifier: Apache-2.0
# TZKV research hook: descriptor / attention-metadata tamper.

from __future__ import annotations

from typing import Any
import torch

from vllm.v1.worker.tzkv_log_utils import env_bool, env_int, write_event


class TZKVDescriptorAttack:
    def __init__(self) -> None:
        self.enabled = env_bool("TZKV_DESC_ATTACK_ENABLE", "0")
        self.max_blocks = env_int("TZKV_DESC_ATTACK_MAX_BLOCKS", 256)
        self.source_order = env_int("TZKV_DESC_ATTACK_SOURCE_ORDER", 0)
        self.victim_order = env_int("TZKV_DESC_ATTACK_VICTIM_ORDER", 1)
        self.attack_once = env_bool("TZKV_DESC_ATTACK_ONCE", "1")

        self._req_order: list[str] = []
        self._req_role: dict[str, str] = {}
        self._source_row: torch.Tensor | None = None
        self._tampered_count = 0

        if self.enabled:
            write_event("desc_attack_hook_init", {
                "max_blocks": self.max_blocks,
                "source_order": self.source_order,
                "victim_order": self.victim_order,
                "attack_once": self.attack_once,
            })

    @staticmethod
    def _req_ids(runner: Any) -> list[str]:
        try:
            return [str(x) for x in list(getattr(runner.input_batch, "req_ids", []) or [])]
        except Exception:
            return []

    def _assign_roles(self, runner: Any) -> None:
        for req_id in self._req_ids(runner):
            if req_id in self._req_role:
                continue

            order = len(self._req_order)
            self._req_order.append(req_id)

            role = "other"
            if order == self.source_order:
                role = "source"
            elif order == self.victim_order:
                role = "victim"

            self._req_role[req_id] = role
            write_event("desc_request_role_assigned", {
                "req_id": req_id,
                "order": order,
                "role": role,
                "active_req_ids": self._req_ids(runner),
            })

    @staticmethod
    def _metadata_bt_tensor(common_attn_metadata: Any) -> torch.Tensor | None:
        for name in ("block_table_tensor", "block_tables", "block_table"):
            x = getattr(common_attn_metadata, name, None)
            if isinstance(x, torch.Tensor):
                return x
        return None

    @staticmethod
    def _set_metadata_bt_tensor(common_attn_metadata: Any, x: torch.Tensor) -> bool:
        for name in ("block_table_tensor", "block_tables", "block_table"):
            if hasattr(common_attn_metadata, name):
                try:
                    setattr(common_attn_metadata, name, x)
                    return True
                except Exception:
                    pass
        return False

    def _active_index(self, runner: Any, req_id: str) -> int | None:
        ids = self._req_ids(runner)
        try:
            return ids.index(req_id)
        except ValueError:
            return None

    def after_common_metadata(
        self,
        runner: Any,
        common_attn_metadata: Any,
        kv_cache_gid: int | None = None,
    ) -> None:
        if not self.enabled:
            return

        try:
            self._assign_roles(runner)

            bt = self._metadata_bt_tensor(common_attn_metadata)
            if bt is None:
                write_event("desc_attack_skip_no_block_table_tensor", {
                    "kv_cache_gid": kv_cache_gid,
                    "metadata_type": type(common_attn_metadata).__name__,
                })
                return
            if bt.ndim < 2:
                write_event("desc_attack_skip_bad_shape", {
                    "shape": tuple(bt.shape),
                    "kv_cache_gid": kv_cache_gid,
                })
                return

            try:
                cloned = bt.clone()
                self._set_metadata_bt_tensor(common_attn_metadata, cloned)
                bt = cloned
            except Exception:
                pass

            for req_id in self._req_ids(runner):
                role = self._req_role.get(req_id, "other")
                row_idx = self._active_index(runner, req_id)
                if row_idx is None or row_idx >= int(bt.shape[0]):
                    continue

                n_blocks = min(self.max_blocks, int(bt.shape[1]))

                if role == "source" and self._source_row is None:
                    self._source_row = bt[row_idx, :n_blocks].detach().clone()
                    valid = self._source_row[self._source_row >= 0]
                    write_event("desc_source_snapshot_ok", {
                        "req_id": req_id,
                        "row_idx": row_idx,
                        "kv_cache_gid": kv_cache_gid,
                        "n_blocks": n_blocks,
                        "valid_blocks": int(valid.numel()),
                        "first_blocks": valid[:16].detach().cpu().tolist(),
                    })

                elif role == "victim":
                    if self.attack_once and self._tampered_count > 0:
                        continue
                    if self._source_row is None:
                        write_event("desc_victim_tamper_skip_no_source", {
                            "req_id": req_id,
                            "kv_cache_gid": kv_cache_gid,
                        })
                        continue

                    n = min(n_blocks, int(self._source_row.numel()))
                    old = bt[row_idx, :n].detach().clone()
                    bt[row_idx, :n] = self._source_row[:n].to(
                        device=bt.device,
                        dtype=bt.dtype,
                        non_blocking=True,
                    )
                    self._tampered_count += 1

                    write_event("desc_victim_tamper_done", {
                        "req_id": req_id,
                        "row_idx": row_idx,
                        "kv_cache_gid": kv_cache_gid,
                        "n_blocks": n,
                        "old_first_blocks": old[old >= 0][:16].detach().cpu().tolist(),
                        "new_first_blocks": bt[row_idx, :n][bt[row_idx, :n] >= 0][:16].detach().cpu().tolist(),
                        "tampered_count": self._tampered_count,
                    })

        except Exception as e:
            write_event("desc_attack_error", {
                "error": repr(e),
                "kv_cache_gid": kv_cache_gid,
            })


# Compatibility with older names.
TZKVDescriptorAttackHook = TZKVDescriptorAttack
