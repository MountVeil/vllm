# SPDX-License-Identifier: Apache-2.0
# TZKV research hook: KV payload tamper.

from __future__ import annotations

import traceback
from typing import Any

import torch

from vllm.v1.worker.tzkv_log_utils import env_bool, env_int, now_ts, write_event


class TZKVKVPayloadAttack:
    def __init__(self) -> None:
        self.enabled = env_bool("TZKV_KV_ATTACK_ENABLE", "0")
        self.max_tokens = env_int("TZKV_KV_ATTACK_MAX_TOKENS", 128)
        self.min_prefill_tokens = env_int("TZKV_KV_ATTACK_MIN_PREFILL", 8)
        self.source_order = env_int("TZKV_KV_ATTACK_SOURCE_ORDER", 0)
        self.victim_order = env_int("TZKV_KV_ATTACK_VICTIM_ORDER", 1)
        self.attack_once = env_bool("TZKV_KV_ATTACK_ONCE", "1")

        self._req_order: list[str] = []
        self._req_role: dict[str, str] = {}
        self._source_snapshot: list[dict[str, Any]] | None = None
        self._tampered_count = 0
        self._last_error_ts = 0.0

        if self.enabled:
            write_event("kv_attack_hook_init", {
                "max_tokens": self.max_tokens,
                "min_prefill_tokens": self.min_prefill_tokens,
                "source_order": self.source_order,
                "victim_order": self.victim_order,
                "attack_once": self.attack_once,
            })

    @staticmethod
    def _req_ids(runner: Any) -> list[str]:
        return [str(x) for x in list(getattr(runner.input_batch, "req_ids", []) or [])]

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
            write_event("kv_request_role_assigned", {
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

    def _req_token_slices(self, runner: Any, scheduler_output: Any) -> dict[str, slice]:
        out: dict[str, slice] = {}
        offset = 0

        for req_id in self._req_ids(runner):
            n = self._scheduled_tokens(scheduler_output, req_id)
            if n <= 0:
                continue
            out[req_id] = slice(offset, offset + n)
            offset += n

        return out

    @staticmethod
    def _extract_one_slot_mapping(slot_mappings: Any) -> torch.Tensor | None:
        if slot_mappings is None:
            return None
        if isinstance(slot_mappings, torch.Tensor):
            return slot_mappings
        if isinstance(slot_mappings, dict):
            for v in slot_mappings.values():
                if isinstance(v, torch.Tensor):
                    return v
            return None
        if isinstance(slot_mappings, (list, tuple)):
            for item in slot_mappings:
                got = TZKVKVPayloadAttack._extract_one_slot_mapping(item)
                if got is not None:
                    return got
        return None

    @staticmethod
    def _cache_layout(cache: torch.Tensor) -> tuple[str, int] | None:
        if not isinstance(cache, torch.Tensor) or cache.ndim < 4:
            return None

        # Layout A: (2, num_blocks, block_size, num_kv_heads, head_size, ...)
        if int(cache.shape[0]) == 2:
            return ("kv_first", int(cache.shape[2]))

        # Layout B: (num_blocks, 2, block_size, num_kv_heads, head_size, ...)
        if cache.ndim >= 5 and int(cache.shape[1]) == 2:
            return ("block_first", int(cache.shape[2]))

        return None

    @staticmethod
    def _select_slots(cache: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
        layout = TZKVKVPayloadAttack._cache_layout(cache)
        if layout is None:
            raise RuntimeError(f"unsupported kv cache shape: {tuple(cache.shape)}")

        kind, block_size = layout
        slots = slots.to(device=cache.device, dtype=torch.long)
        block_ids = torch.div(slots, block_size, rounding_mode="floor")
        offsets = slots.remainder(block_size)

        if kind == "kv_first":
            return cache[(slice(None), block_ids, offsets, Ellipsis)]
        return cache[(block_ids, slice(None), offsets, Ellipsis)]

    @staticmethod
    def _assign_slots(cache: torch.Tensor, slots: torch.Tensor, value: torch.Tensor) -> None:
        layout = TZKVKVPayloadAttack._cache_layout(cache)
        if layout is None:
            raise RuntimeError(f"unsupported kv cache shape: {tuple(cache.shape)}")

        kind, block_size = layout
        slots = slots.to(device=cache.device, dtype=torch.long)
        block_ids = torch.div(slots, block_size, rounding_mode="floor")
        offsets = slots.remainder(block_size)

        if kind == "kv_first":
            cache[(slice(None), block_ids, offsets, Ellipsis)] = value.to(
                device=cache.device,
                dtype=cache.dtype,
                non_blocking=True,
            )
        else:
            cache[(block_ids, slice(None), offsets, Ellipsis)] = value.to(
                device=cache.device,
                dtype=cache.dtype,
                non_blocking=True,
            )

    def _valid_slots(self, slot_mapping: torch.Tensor, req_slice: slice) -> torch.Tensor:
        slots = slot_mapping[req_slice].detach()
        slots = slots[slots >= 0]
        if slots.numel() > self.max_tokens:
            slots = slots[: self.max_tokens]
        return slots.to(dtype=torch.long)

    def _snapshot_source(self, runner: Any, slots: torch.Tensor, req_id: str) -> None:
        if slots.numel() < self.min_prefill_tokens:
            write_event("kv_source_snapshot_skip_small_prefill", {
                "req_id": req_id,
                "num_slots": int(slots.numel()),
            })
            return

        snapshot: list[dict[str, Any]] = []
        kv_caches = list(getattr(runner, "kv_caches", []) or [])

        for layer_idx, cache in enumerate(kv_caches):
            if not isinstance(cache, torch.Tensor):
                continue

            layout = self._cache_layout(cache)
            if layout is None:
                write_event("kv_source_snapshot_skip_layer", {
                    "layer_idx": layer_idx,
                    "shape": tuple(cache.shape),
                    "reason": "unsupported_shape",
                })
                continue

            try:
                value = self._select_slots(cache, slots).detach().clone()
                snapshot.append({
                    "layer_idx": layer_idx,
                    "layout": layout[0],
                    "block_size": layout[1],
                    "shape": tuple(cache.shape),
                    "value": value,
                })
            except Exception as e:
                write_event("kv_source_snapshot_layer_error", {
                    "layer_idx": layer_idx,
                    "shape": tuple(cache.shape),
                    "error": repr(e),
                })

        if not snapshot:
            write_event("kv_source_snapshot_fail", {
                "req_id": req_id,
                "reason": "no_supported_layers",
            })
            return

        try:
            torch.cuda.synchronize()
        except Exception:
            pass

        self._source_snapshot = snapshot
        write_event("kv_source_snapshot_ok", {
            "req_id": req_id,
            "num_slots": int(slots.numel()),
            "num_layers": len(snapshot),
            "first_layers": [
                {
                    "layer_idx": x["layer_idx"],
                    "shape": x["shape"],
                    "layout": x["layout"],
                    "block_size": x["block_size"],
                    "value_shape": tuple(x["value"].shape),
                }
                for x in snapshot[:4]
            ],
        })

    def _tamper_victim(self, runner: Any, slots: torch.Tensor, req_id: str) -> None:
        if self.attack_once and self._tampered_count > 0:
            return

        if self._source_snapshot is None:
            write_event("kv_victim_tamper_skip_no_source", {"req_id": req_id})
            return

        if slots.numel() < self.min_prefill_tokens:
            write_event("kv_victim_tamper_skip_small_prefill", {
                "req_id": req_id,
                "num_slots": int(slots.numel()),
            })
            return

        kv_caches = list(getattr(runner, "kv_caches", []) or [])
        copied_layers = 0
        n_dst = int(slots.numel())

        for snap in self._source_snapshot:
            layer_idx = int(snap["layer_idx"])
            if layer_idx >= len(kv_caches):
                continue

            cache = kv_caches[layer_idx]
            if not isinstance(cache, torch.Tensor):
                continue

            value: torch.Tensor = snap["value"]
            if snap["layout"] == "kv_first":
                n = min(n_dst, int(value.shape[1]))
                src_value = value[:, :n, ...]
            else:
                n = min(n_dst, int(value.shape[0]))
                src_value = value[:n, ...]

            if n <= 0:
                continue

            try:
                self._assign_slots(cache, slots[:n], src_value)
                copied_layers += 1
            except Exception as e:
                write_event("kv_victim_tamper_layer_error", {
                    "req_id": req_id,
                    "layer_idx": layer_idx,
                    "shape": tuple(cache.shape),
                    "error": repr(e),
                })

        try:
            torch.cuda.synchronize()
        except Exception:
            pass

        self._tampered_count += 1
        write_event("kv_victim_tamper_done", {
            "req_id": req_id,
            "num_dst_slots": int(slots.numel()),
            "copied_layers": copied_layers,
            "tampered_count": self._tampered_count,
        })

    @torch.inference_mode()
    def after_model_forward(
        self,
        runner: Any,
        scheduler_output: Any,
        slot_mappings: Any,
        num_tokens: int | None = None,
    ) -> None:
        if not self.enabled:
            return

        try:
            self._assign_roles(runner)

            one_slot_mapping = self._extract_one_slot_mapping(slot_mappings)
            if one_slot_mapping is None:
                write_event("kv_attack_skip_no_slot_mapping", {})
                return

            req_slices = self._req_token_slices(runner, scheduler_output)

            for req_id, req_slice in req_slices.items():
                role = self._req_role.get(req_id, "other")
                n_sched = self._scheduled_tokens(scheduler_output, req_id)

                if n_sched < self.min_prefill_tokens:
                    continue

                slots = self._valid_slots(one_slot_mapping, req_slice)

                if role == "source" and self._source_snapshot is None:
                    self._snapshot_source(runner, slots, req_id)
                elif role == "victim":
                    self._tamper_victim(runner, slots, req_id)

        except Exception as e:
            now = now_ts()
            if now - self._last_error_ts > 2.0:
                self._last_error_ts = now
                write_event("kv_attack_error", {
                    "error": repr(e),
                    "traceback": traceback.format_exc(),
                })
