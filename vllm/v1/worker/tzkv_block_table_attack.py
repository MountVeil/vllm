# SPDX-License-Identifier: Apache-2.0
# TZKV research hook: block-table metadata tamper.

from __future__ import annotations

from typing import Any
import torch

from vllm.v1.worker.tzkv_log_utils import env_bool, env_int, write_event


class TZKVBlockTableAttack:
    def __init__(self) -> None:
        self.enabled = env_bool("TZKV_BT_ATTACK_ENABLE", "0")
        self.max_blocks = env_int("TZKV_BT_ATTACK_MAX_BLOCKS", 256)
        self.min_prefill_tokens = env_int("TZKV_BT_ATTACK_MIN_PREFILL", 8)
        self.source_order = env_int("TZKV_BT_ATTACK_SOURCE_ORDER", 0)
        self.victim_order = env_int("TZKV_BT_ATTACK_VICTIM_ORDER", 1)
        self.attack_once = env_bool("TZKV_BT_ATTACK_ONCE", "1")

        self._req_order: list[str] = []
        self._req_role: dict[str, str] = {}
        self._source_row: torch.Tensor | None = None
        self._tampered_count = 0

        if self.enabled:
            write_event("bt_attack_hook_init", {
                "max_blocks": self.max_blocks,
                "min_prefill_tokens": self.min_prefill_tokens,
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
            write_event("bt_request_role_assigned", {
                "req_id": req_id,
                "order": order,
                "role": role,
                "active_req_ids": self._req_ids(runner),
            })

    @staticmethod
    def _scheduled_tokens(scheduler_output: Any, req_id: str) -> int:
        obj = getattr(scheduler_output, "num_scheduled_tokens", None)
        if obj is None:
            return 0
        for key in (req_id, str(req_id)):
            try:
                return int(obj.get(key, 0))
            except Exception:
                pass
        try:
            return int(obj[req_id])
        except Exception:
            return 0

    @staticmethod
    def _bt_tensor(runner: Any) -> torch.Tensor | None:
        try:
            bt = getattr(runner.input_batch, "block_table", None)
        except Exception:
            return None
        if bt is None:
            return None

        for name in ("gpu", "block_table", "block_table_tensor"):
            x = getattr(bt, name, None)
            if isinstance(x, torch.Tensor):
                return x
        return None

    def _active_index(self, runner: Any, req_id: str) -> int | None:
        ids = self._req_ids(runner)
        try:
            return ids.index(req_id)
        except ValueError:
            return None

    def after_batch_update(self, runner: Any, scheduler_output: Any) -> None:
        if not self.enabled:
            return

        try:
            self._assign_roles(runner)

            bt = self._bt_tensor(runner)
            if bt is None:
                write_event("bt_attack_skip_no_tensor", {})
                return
            if bt.ndim < 2:
                write_event("bt_attack_skip_bad_shape", {"shape": tuple(bt.shape)})
                return

            for req_id in self._req_ids(runner):
                role = self._req_role.get(req_id, "other")
                n_sched = self._scheduled_tokens(scheduler_output, req_id)
                if n_sched < self.min_prefill_tokens:
                    continue

                row_idx = self._active_index(runner, req_id)
                if row_idx is None or row_idx >= int(bt.shape[0]):
                    continue

                n_blocks = min(self.max_blocks, int(bt.shape[1]))

                if role == "source" and self._source_row is None:
                    self._source_row = bt[row_idx, :n_blocks].detach().clone()
                    valid = self._source_row[self._source_row >= 0]
                    write_event("bt_source_snapshot_ok", {
                        "req_id": req_id,
                        "row_idx": row_idx,
                        "n_blocks": n_blocks,
                        "valid_blocks": int(valid.numel()),
                        "first_blocks": valid[:16].detach().cpu().tolist(),
                    })

                elif role == "victim":
                    if self.attack_once and self._tampered_count > 0:
                        continue
                    if self._source_row is None:
                        write_event("bt_victim_tamper_skip_no_source", {"req_id": req_id})
                        continue

                    n = min(n_blocks, int(self._source_row.numel()))
                    old = bt[row_idx, :n].detach().clone()
                    bt[row_idx, :n] = self._source_row[:n].to(
                        device=bt.device,
                        dtype=bt.dtype,
                        non_blocking=True,
                    )
                    self._tampered_count += 1

                    write_event("bt_victim_tamper_done", {
                        "req_id": req_id,
                        "row_idx": row_idx,
                        "n_blocks": n,
                        "old_first_blocks": old[old >= 0][:16].detach().cpu().tolist(),
                        "new_first_blocks": bt[row_idx, :n][bt[row_idx, :n] >= 0][:16].detach().cpu().tolist(),
                        "tampered_count": self._tampered_count,
                    })

        except Exception as e:
            write_event("bt_attack_error", {"error": repr(e)})


    def after_slot_mapping_commit(self, *args: Any, **kwargs: Any) -> None:
        """Compatibility wrapper for older gpu_model_runner.py patches.

        Older TZKV BT attack patches called this method from _prepare_inputs()
        after slot mapping was committed. The current unified attack uses
        after_batch_update(runner, scheduler_output). This wrapper prevents
        stale call sites from killing the engine, and when enough context is
        available it forwards to after_batch_update().
        """
        if not self.enabled:
            return

        try:
            runner = (
                kwargs.get("runner")
                or kwargs.get("model_runner")
                or kwargs.get("gpu_model_runner")
            )
            scheduler_output = kwargs.get("scheduler_output")

            # Common fallback: some old call sites pass runner as the first arg.
            if runner is None and args:
                cand = args[0]
                if hasattr(cand, "input_batch"):
                    runner = cand

            if runner is not None and scheduler_output is not None:
                self.after_batch_update(
                    runner=runner,
                    scheduler_output=scheduler_output,
                )
                return

            write_event("bt_after_slot_mapping_commit_skip_no_context", {
                "num_args": len(args),
                "kw_keys": sorted(list(kwargs.keys())),
                "has_runner": runner is not None,
                "has_scheduler_output": scheduler_output is not None,
            })
        except Exception as e:
            write_event("bt_after_slot_mapping_commit_error", {
                "error": repr(e),
            })


# Compatibility with older names.
TZKVBlockTableAttackHook = TZKVBlockTableAttack
