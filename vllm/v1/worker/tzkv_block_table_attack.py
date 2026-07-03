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


# TZKV_BT_TENSOR_READY_METHOD_BEGIN
def _tzkv_bt_after_block_table_tensor_ready(
    self,
    *,
    runner,
    scheduler_output=None,
    block_table_tensor=None,
    slot_mapping=None,
    num_tokens=0,
):
    """Attack the actual block_table_tensor passed into attention metadata.

    This path is intentionally close to CommonAttentionMetadata construction:
    block_table_gid_0 -> common_attn_metadata.block_table_tensor.
    """
    try:
        from vllm.v1.worker.tzkv_log_utils import write_event
    except Exception:
        write_event = None

    def log(event, obj=None):
        if write_event is not None:
            try:
                write_event(event, obj or {})
            except Exception:
                pass

    if not getattr(self, "enabled", False):
        return

    if block_table_tensor is None:
        log("bt_attack_skip_no_block_table_tensor")
        return

    try:
        import torch

        if not torch.is_tensor(block_table_tensor):
            log("bt_attack_skip_non_tensor", {
                "type": type(block_table_tensor).__name__,
            })
            return

        if block_table_tensor.ndim < 2:
            log("bt_attack_skip_bad_shape", {
                "shape": list(block_table_tensor.shape),
            })
            return

        # In this workload, requests are sequential. The active row is usually 0.
        req_ids = []
        try:
            req_ids = list(getattr(getattr(runner, "input_batch", None), "req_ids", []))
        except Exception:
            req_ids = []

        active_rows = int(block_table_tensor.shape[0])
        row_idx = 0
        if active_rows <= 0:
            log("bt_attack_skip_empty_tensor", {
                "shape": list(block_table_tensor.shape),
            })
            return

        # Identify request identity if possible. Fallback to monotonically assigned roles.
        current_req = None
        for x in req_ids:
            if x is not None:
                current_req = str(x)
                break
        if current_req is None:
            current_req = f"anon_call_{getattr(self, '_tzkv_bt_call_idx', 0)}"

        seen = getattr(self, "_tzkv_bt_seen_reqs", [])
        if current_req not in seen:
            seen.append(current_req)
            setattr(self, "_tzkv_bt_seen_reqs", seen)
        order = seen.index(current_req)

        source_order = int(getattr(self, "source_order", 0))
        victim_order = int(getattr(self, "victim_order", 1))
        attack_once = bool(getattr(self, "attack_once", True))

        if order == source_order:
            # Snapshot source logical->physical block-table row.
            snap = block_table_tensor[row_idx].detach().clone()
            setattr(self, "_tzkv_bt_source_snapshot", snap)
            setattr(self, "_tzkv_bt_source_req", current_req)
            log("bt_source_snapshot_ok", {
                "req_id": current_req,
                "row_idx": row_idx,
                "shape": list(block_table_tensor.shape),
                "snapshot_numel": int(snap.numel()),
                "num_tokens": int(num_tokens or 0),
            })
            return

        if order == victim_order:
            if attack_once and getattr(self, "_tzkv_bt_done", False):
                log("bt_attack_skip_once_done", {
                    "req_id": current_req,
                })
                return

            snap = getattr(self, "_tzkv_bt_source_snapshot", None)
            if snap is None:
                log("bt_attack_skip_no_source_snapshot", {
                    "req_id": current_req,
                    "seen": seen,
                })
                return

            dst = block_table_tensor[row_idx]
            n = min(int(dst.numel()), int(snap.numel()))
            if n <= 0:
                log("bt_attack_skip_zero_len", {
                    "req_id": current_req,
                    "dst_numel": int(dst.numel()),
                    "snap_numel": int(snap.numel()),
                })
                return

            # Mutate the actual tensor that will be embedded in attention metadata.
            dst.view(-1)[:n].copy_(snap.view(-1)[:n].to(device=dst.device, dtype=dst.dtype))
            setattr(self, "_tzkv_bt_done", True)

            log("bt_victim_tamper_done", {
                "victim_req_id": current_req,
                "source_req_id": getattr(self, "_tzkv_bt_source_req", None),
                "row_idx": row_idx,
                "copied_entries": int(n),
                "shape": list(block_table_tensor.shape),
                "num_tokens": int(num_tokens or 0),
            })
            return

        log("bt_attack_skip_role", {
            "req_id": current_req,
            "order": int(order),
            "source_order": source_order,
            "victim_order": victim_order,
            "seen": seen,
            "shape": list(block_table_tensor.shape),
        })

    except Exception as e:
        log("bt_attack_error", {
            "where": "after_block_table_tensor_ready",
            "error": repr(e),
        })


try:
    TZKVBlockTableAttack.after_block_table_tensor_ready = _tzkv_bt_after_block_table_tensor_ready
except NameError:
    pass

try:
    TZKVBlockTableAttackHook.after_block_table_tensor_ready = _tzkv_bt_after_block_table_tensor_ready
except NameError:
    pass
# TZKV_BT_TENSOR_READY_METHOD_END
