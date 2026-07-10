"""Fixed-schema vLLM runtime-state hooks for TZKV.

Only explicit, versioned fields are hashed. This module never recursively
serializes attention backend objects and does not depend on a TA-specific ABI.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tzkv_runtime import (
    BlockTableApproval as BackendBlockTableApproval,
    DescriptorApproval as BackendDescriptorApproval,
    SlotMappingApproval as BackendSlotMappingApproval,
    TZKVError as BackendTZKVError,
    UseTicket as BackendUseTicket,
    get_runtime_backend,
)


DESCRIPTOR_SCHEMA_VERSION = 1
BLOCK_TABLE_SCHEMA_VERSION = 1


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _tensor_to_numpy(value: torch.Tensor) -> np.ndarray:
    tensor = value.detach()
    if tensor.is_cuda:
        tensor = tensor.cpu()
    return tensor.contiguous().numpy()


def _hash_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)

    h = hashlib.sha256()
    h.update(b"TZKV.array.v1\0")
    h.update(str(array.dtype).encode("ascii"))
    h.update(b"\0")
    h.update(
        json.dumps(
            list(array.shape),
            separators=(",", ":"),
        ).encode("ascii")
    )
    h.update(b"\0")
    h.update(array.tobytes(order="C"))
    return h.hexdigest()


def _optional_tensor_root(
    value: torch.Tensor | None,
    *,
    length: int | None = None,
) -> str | None:
    if value is None:
        return None

    tensor = value if length is None else value[:length]
    return _hash_array(_tensor_to_numpy(tensor))


def _encode_u64(h: "hashlib._Hash", value: int) -> None:
    h.update(int(value).to_bytes(8, "little", signed=False))


def _encode_bool(h: "hashlib._Hash", value: bool) -> None:
    h.update(b"\x01" if value else b"\x00")


def _encode_text(h: "hashlib._Hash", value: str) -> None:
    encoded = value.encode("utf-8", "replace")
    _encode_u64(h, len(encoded))
    h.update(encoded)


def _encode_optional_root(
    h: "hashlib._Hash",
    value: str | None,
) -> None:
    if value is None:
        h.update(b"\x00")
        return

    h.update(b"\x01")
    h.update(bytes.fromhex(value))


@dataclass(frozen=True)
class BlockTableApproval:
    batch_id: int
    num_reqs: int
    request_ids: tuple[str, ...]
    group_roots: tuple[str, ...]
    row_lengths: tuple[tuple[int, ...], ...]
    backend_approval: BackendBlockTableApproval


@dataclass(frozen=True)
class SlotMappingApproval:
    batch_id: int
    block_table_roots: tuple[str, ...]
    slot_mapping_roots: tuple[str, ...]
    num_tokens: int
    backend_approval: BackendSlotMappingApproval


@dataclass(frozen=True)
class DescriptorApproval:
    schema_version: int
    batch_id: int
    descriptor_root: str
    num_reqs: int
    num_reqs_padded: int
    num_tokens: int
    num_tokens_padded: int
    cudagraph_mode: str
    record_count: int
    backend_approval: BackendDescriptorApproval
    use_ticket: BackendUseTicket


class TZKVRuntimeHooks:
    def __init__(self) -> None:
        self.enabled = _env_bool("TZKV_HOOK_ENABLE", False)
        self.strict = _env_bool("TZKV_HOOK_STRICT", True)
        self.log_full_values = _env_bool(
            "TZKV_HOOK_LOG_FULL_VALUES",
            False,
        )
        self.log_path = Path(
            os.getenv(
                "TZKV_HOOK_LOG",
                "logs/tzkv_runtime_hooks.jsonl",
            )
        )

        self._lock = threading.Lock()
        self._next_batch_id = 1
        self.backend = (
            get_runtime_backend()
            if self.enabled
            else None
        )

    def _new_batch_id(self) -> int:
        with self._lock:
            batch_id = self._next_batch_id
            self._next_batch_id += 1
            return batch_id

    def _event(self, event: str, **fields: Any) -> None:
        if not self.enabled:
            return

        row = {
            "timestamp_ns": time.time_ns(),
            "event": event,
            **fields,
        }

        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        with self._lock:
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        row,
                        sort_keys=True,
                        default=str,
                    )
                    + "\n"
                )

    def _fail(self, message: str, **fields: Any) -> bool:
        self._event(
            "hook_verify_failure",
            message=message,
            **fields,
        )

        if self.strict:
            raise RuntimeError(
                f"TZKV runtime hook: {message}"
            )

        return False

    @staticmethod
    def _tables(block_table: Any) -> list[Any]:
        if hasattr(block_table, "block_tables"):
            return list(block_table.block_tables)
        return [block_table]

    @staticmethod
    def _request_ids(
        runner: Any,
        num_reqs: int,
    ) -> tuple[str, ...]:
        req_ids = tuple(
            str(x)
            for x in runner.input_batch.req_ids[:num_reqs]
        )

        if len(req_ids) != num_reqs:
            raise RuntimeError(
                "request-row mismatch: "
                f"num_reqs={num_reqs}, "
                f"req_ids={len(req_ids)}"
            )

        return req_ids

    def _block_table_snapshot(
        self,
        block_table: Any,
        num_reqs: int,
    ) -> tuple[
        tuple[str, ...],
        tuple[tuple[int, ...], ...],
        list[dict[str, Any]],
    ]:
        roots: list[str] = []
        all_lengths: list[tuple[int, ...]] = []
        groups: list[dict[str, Any]] = []

        for group_id, table in enumerate(
            self._tables(block_table)
        ):
            lengths = tuple(
                int(x)
                for x in table.num_blocks_per_row[
                    :num_reqs
                ].tolist()
            )

            rows: list[list[int]] = []

            h = hashlib.sha256()
            h.update(b"TZKV.vllm.block_table.v1\0")
            _encode_u64(h, BLOCK_TABLE_SCHEMA_VERSION)
            _encode_u64(h, group_id)
            _encode_u64(h, num_reqs)

            for row_idx, length in enumerate(lengths):
                if (
                    length < 0
                    or length > table.max_num_blocks_per_req
                ):
                    raise RuntimeError(
                        "invalid block-table row length: "
                        f"group={group_id}, "
                        f"row={row_idx}, "
                        f"length={length}"
                    )

                row = np.ascontiguousarray(
                    table.block_table.np[
                        row_idx,
                        :length,
                    ],
                    dtype=np.int32,
                )

                _encode_u64(h, row_idx)
                _encode_u64(h, length)
                h.update(row.tobytes(order="C"))

                if self.log_full_values:
                    rows.append(
                        [int(x) for x in row.tolist()]
                    )

            root = h.hexdigest()
            roots.append(root)
            all_lengths.append(lengths)

            record: dict[str, Any] = {
                "group_id": group_id,
                "root": root,
                "row_lengths": list(lengths),
            }

            if self.log_full_values:
                record["rows"] = rows

            groups.append(record)

        return (
            tuple(roots),
            tuple(all_lengths),
            groups,
        )

    def on_block_table_ready(
        self,
        *,
        runner: Any,
        scheduler_output: Any,
        block_table: Any,
        num_reqs: int,
    ) -> BlockTableApproval | None:
        if not self.enabled:
            return None

        request_ids = self._request_ids(
            runner,
            num_reqs,
        )

        roots, row_lengths, groups = (
            self._block_table_snapshot(
                block_table,
                num_reqs,
            )
        )

        assert self.backend is not None

        try:
            backend_approval = (
                self.backend.approve_block_table(
                    num_reqs=num_reqs,
                    request_ids=request_ids,
                    group_roots=tuple(
                        bytes.fromhex(root)
                        for root in roots
                    ),
                    row_lengths=row_lengths,
                )
            )
        except BackendTZKVError as exc:
            self._fail(
                "backend rejected block-table approval",
                error=str(exc),
            )
            return None

        batch_id = backend_approval.step_id

        approval = BlockTableApproval(
            batch_id=batch_id,
            num_reqs=num_reqs,
            request_ids=request_ids,
            group_roots=roots,
            row_lengths=row_lengths,
            backend_approval=backend_approval,
        )

        self._event(
            "block_table_ready",
            schema_version=BLOCK_TABLE_SCHEMA_VERSION,
            batch_id=batch_id,
            num_reqs=num_reqs,
            request_ids=request_ids,
            total_num_scheduled_tokens=int(
                scheduler_output.total_num_scheduled_tokens
            ),
            groups=groups,
        )

        return approval

    def on_block_table_committed(
        self,
        *,
        approval: BlockTableApproval | None,
        runner: Any,
        block_table: Any,
        num_reqs: int,
    ) -> bool:
        if not self.enabled:
            return True

        if approval is None:
            return self._fail(
                "missing block-table approval"
            )

        request_ids = self._request_ids(
            runner,
            num_reqs,
        )

        roots, row_lengths, _ = (
            self._block_table_snapshot(
                block_table,
                num_reqs,
            )
        )

        if num_reqs != approval.num_reqs:
            return self._fail(
                "num_reqs changed during block-table commit"
            )

        if request_ids != approval.request_ids:
            return self._fail(
                "request-row mapping changed during "
                "block-table commit"
            )

        if roots != approval.group_roots:
            return self._fail(
                "CPU block table changed during GPU commit"
            )

        if row_lengths != approval.row_lengths:
            return self._fail(
                "block-table row lengths changed during "
                "GPU commit"
            )

        # Compare exactly the effective row ranges included
        # in the approved block-table digest.
        for group_id, table in enumerate(
            self._tables(block_table)
        ):
            lengths = approval.row_lengths[group_id]

            for row_idx, length in enumerate(lengths):
                cpu_row = np.ascontiguousarray(
                    table.block_table.np[
                        row_idx,
                        :length,
                    ],
                    dtype=np.int32,
                )

                gpu_row = _tensor_to_numpy(
                    table.block_table.gpu[
                        row_idx,
                        :length,
                    ]
                ).astype(
                    np.int32,
                    copy=False,
                )

                if (
                    cpu_row.shape != gpu_row.shape
                    or not np.array_equal(
                        cpu_row,
                        gpu_row,
                    )
                ):
                    return self._fail(
                        "GPU block-table row differs "
                        "from approved CPU row",
                        group_id=group_id,
                        row_idx=row_idx,
                        row_length=length,
                    )

        assert self.backend is not None

        try:
            self.backend.verify_block_table_commit(
                approval.backend_approval,
                num_reqs=num_reqs,
                request_ids=request_ids,
                group_roots=tuple(
                    bytes.fromhex(root)
                    for root in roots
                ),
                row_lengths=row_lengths,
            )
        except BackendTZKVError as exc:
            return self._fail(
                "backend rejected block-table commit",
                error=str(exc),
            )

        self._event(
            "block_table_committed",
            batch_id=approval.batch_id,
            group_roots=approval.group_roots,
        )

        return True

    def on_slot_mapping_ready(
        self,
        *,
        block_table_approval: BlockTableApproval | None,
        runner: Any,
        scheduler_output: Any,
        block_table: Any,
        req_indices: np.ndarray,
        positions: np.ndarray,
        num_tokens: int,
    ) -> SlotMappingApproval | None:
        if not self.enabled:
            return None

        if block_table_approval is None:
            self._fail(
                "slot mapping generated without "
                "block-table approval"
            )
            return None

        slot_roots: list[str] = []
        groups: list[dict[str, Any]] = []

        for group_id, table in enumerate(
            self._tables(block_table)
        ):
            cpu_slots = np.ascontiguousarray(
                table.slot_mapping.np[:num_tokens],
                dtype=np.int64,
            )

            gpu_slots = _tensor_to_numpy(
                table.slot_mapping.gpu[:num_tokens]
            ).astype(
                np.int64,
                copy=False,
            )

            if not np.array_equal(
                cpu_slots,
                gpu_slots,
            ):
                self._fail(
                    "GPU slot mapping differs from "
                    "CPU slot mapping",
                    group_id=group_id,
                )

            root = _hash_array(cpu_slots)
            slot_roots.append(root)

            record: dict[str, Any] = {
                "group_id": group_id,
                "slot_mapping_root": root,
            }

            if self.log_full_values:
                record["slot_mapping"] = [
                    int(x)
                    for x in cpu_slots.tolist()
                ]

            groups.append(record)

        assert self.backend is not None

        try:
            backend_approval = (
                self.backend.approve_slot_mapping(
                    block_table_approval.backend_approval,
                    num_tokens=num_tokens,
                    slot_mapping_roots=tuple(
                        bytes.fromhex(root)
                        for root in slot_roots
                    ),
                )
            )
        except BackendTZKVError as exc:
            self._fail(
                "backend rejected slot-mapping approval",
                error=str(exc),
            )
            return None

        approval = SlotMappingApproval(
            batch_id=block_table_approval.batch_id,
            block_table_roots=(
                block_table_approval.group_roots
            ),
            slot_mapping_roots=tuple(slot_roots),
            num_tokens=num_tokens,
            backend_approval=backend_approval,
        )

        self._event(
            "slot_mapping_ready",
            batch_id=approval.batch_id,
            num_tokens=num_tokens,
            groups=groups,
        )

        return approval

    def _descriptor_record(
        self,
        *,
        record: dict[str, Any],
        block_table_approval: BlockTableApproval,
        slot_mapping_approval: SlotMappingApproval,
        runner: Any,
    ) -> dict[str, Any]:
        group_id = int(record["kv_cache_group_id"])
        attn_group_id = int(record["attention_group_id"])
        ubatch_id = int(record["ubatch_id"])

        if group_id < 0:
            raise RuntimeError(
                f"invalid kv_cache_group_id: {group_id}"
            )

        if group_id >= len(
            block_table_approval.group_roots
        ):
            raise RuntimeError(
                "descriptor references unknown "
                f"KV-cache group {group_id}"
            )

        block_table_tensor = record["block_table_tensor"]
        slot_mapping = record["slot_mapping"]

        if not isinstance(
            block_table_tensor,
            torch.Tensor,
        ):
            raise TypeError(
                "descriptor block_table_tensor "
                "is not a torch.Tensor"
            )

        if not isinstance(
            slot_mapping,
            torch.Tensor,
        ):
            raise TypeError(
                "descriptor slot_mapping "
                "is not a torch.Tensor"
            )

        table = self._tables(
            runner.input_batch.block_table
        )[group_id]

        row_lengths = (
            block_table_approval.row_lengths[group_id]
        )

        # Final-use BT check: recompute the effective GPU block-table root
        # and compare it with the root approved before the GPU copy.
        gpu_bt_hash = hashlib.sha256()
        gpu_bt_hash.update(b"TZKV.vllm.block_table.v1\0")
        _encode_u64(gpu_bt_hash, BLOCK_TABLE_SCHEMA_VERSION)
        _encode_u64(gpu_bt_hash, group_id)
        _encode_u64(gpu_bt_hash, block_table_approval.num_reqs)

        for row_idx, row_length in enumerate(row_lengths):
            gpu_row = _tensor_to_numpy(
                table.block_table.gpu[
                    row_idx,
                    :row_length,
                ]
            ).astype(
                np.int32,
                copy=False,
            )
            gpu_row = np.ascontiguousarray(
                gpu_row,
                dtype=np.int32,
            )

            _encode_u64(gpu_bt_hash, row_idx)
            _encode_u64(gpu_bt_hash, row_length)
            gpu_bt_hash.update(gpu_row.tobytes(order="C"))

        current_gpu_bt_root = gpu_bt_hash.hexdigest()
        approved_gpu_bt_root = (
            block_table_approval.group_roots[group_id]
        )

        if current_gpu_bt_root != approved_gpu_bt_root:
            self._fail(
                "GPU block table changed before descriptor use",
                kv_cache_group_id=group_id,
                attention_group_id=attn_group_id,
                ubatch_id=ubatch_id,
                expected=approved_gpu_bt_root,
                actual=current_gpu_bt_root,
            )

        # Verify each descriptor row against the now-validated GPU BT.
        for row_idx, row_length in enumerate(row_lengths):
            descriptor_row = _tensor_to_numpy(
                block_table_tensor[
                    row_idx,
                    :row_length,
                ]
            ).astype(
                np.int32,
                copy=False,
            )

            approved_row = _tensor_to_numpy(
                table.block_table.gpu[
                    row_idx,
                    :row_length,
                ]
            ).astype(
                np.int32,
                copy=False,
            )

            if not np.array_equal(
                descriptor_row,
                approved_row,
            ):
                self._fail(
                    "descriptor block table differs "
                    "from approved block table",
                    kv_cache_group_id=group_id,
                    attention_group_id=attn_group_id,
                    ubatch_id=ubatch_id,
                    row_idx=row_idx,
                    row_length=row_length,
                )

        num_effective_tokens = (
            slot_mapping_approval.num_tokens
        )

        descriptor_slots = _tensor_to_numpy(
            slot_mapping[:num_effective_tokens]
        ).astype(
            np.int64,
            copy=False,
        )

        approved_slots = _tensor_to_numpy(
            table.slot_mapping.gpu[
                :num_effective_tokens
            ]
        ).astype(
            np.int64,
            copy=False,
        )

        if not np.array_equal(
            descriptor_slots,
            approved_slots,
        ):
            self._fail(
                "descriptor slot mapping differs "
                "from approved slot mapping",
                kv_cache_group_id=group_id,
                attention_group_id=attn_group_id,
                ubatch_id=ubatch_id,
            )

        query_start_loc = record["query_start_loc"]
        seq_lens = record["seq_lens"]
        num_computed_tokens = record[
            "num_computed_tokens"
        ]

        fixed = {
            "kv_cache_group_id": group_id,
            "attention_group_id": attn_group_id,
            "ubatch_id": ubatch_id,
            "num_reqs": int(record["num_reqs"]),
            "num_actual_tokens": int(
                record["num_actual_tokens"]
            ),
            "max_query_len": int(
                record["max_query_len"]
            ),
            "max_seq_len": int(record["max_seq_len"]),
            "causal": bool(record["causal"]),
            "block_table_root": (
                block_table_approval.group_roots[
                    group_id
                ]
            ),
            "slot_mapping_root": (
                slot_mapping_approval.slot_mapping_roots[
                    group_id
                ]
            ),
            "query_start_loc_root": (
                _optional_tensor_root(
                    query_start_loc,
                )
            ),
            "seq_lens_root": _optional_tensor_root(
                seq_lens,
            ),
            "num_computed_tokens_root": (
                _optional_tensor_root(
                    num_computed_tokens,
                )
            ),
            "encoder_seq_lens_root": (
                _optional_tensor_root(
                    record.get("encoder_seq_lens")
                )
            ),
            "dcp_local_seq_lens_root": (
                _optional_tensor_root(
                    record.get("dcp_local_seq_lens")
                )
            ),
        }

        return fixed

    @staticmethod
    def _descriptor_root(
        *,
        batch_id: int,
        num_reqs: int,
        num_reqs_padded: int,
        num_tokens: int,
        num_tokens_padded: int,
        cudagraph_mode: str,
        records: list[dict[str, Any]],
    ) -> str:
        h = hashlib.sha256()
        h.update(b"TZKV.vllm.descriptor.v1\0")

        _encode_u64(h, DESCRIPTOR_SCHEMA_VERSION)
        _encode_u64(h, batch_id)
        _encode_u64(h, num_reqs)
        _encode_u64(h, num_reqs_padded)
        _encode_u64(h, num_tokens)
        _encode_u64(h, num_tokens_padded)
        _encode_text(h, cudagraph_mode)
        _encode_u64(h, len(records))

        ordered = sorted(
            records,
            key=lambda x: (
                x["kv_cache_group_id"],
                x["attention_group_id"],
                x["ubatch_id"],
            ),
        )

        for record in ordered:
            _encode_u64(
                h,
                record["kv_cache_group_id"],
            )
            _encode_u64(
                h,
                record["attention_group_id"],
            )

            # -1 denotes no microbatch.
            _encode_u64(
                h,
                record["ubatch_id"] + 1,
            )

            _encode_u64(h, record["num_reqs"])
            _encode_u64(
                h,
                record["num_actual_tokens"],
            )
            _encode_u64(
                h,
                record["max_query_len"],
            )
            _encode_u64(
                h,
                record["max_seq_len"],
            )
            _encode_bool(h, record["causal"])

            h.update(
                bytes.fromhex(
                    record["block_table_root"]
                )
            )
            h.update(
                bytes.fromhex(
                    record["slot_mapping_root"]
                )
            )

            _encode_optional_root(
                h,
                record["query_start_loc_root"],
            )
            _encode_optional_root(
                h,
                record["seq_lens_root"],
            )
            _encode_optional_root(
                h,
                record[
                    "num_computed_tokens_root"
                ],
            )
            _encode_optional_root(
                h,
                record["encoder_seq_lens_root"],
            )
            _encode_optional_root(
                h,
                record[
                    "dcp_local_seq_lens_root"
                ],
            )

        return h.hexdigest()

    def _fixed_descriptor(
        self,
        *,
        descriptor_records: list[dict[str, Any]],
        block_table_approval: BlockTableApproval,
        slot_mapping_approval: SlotMappingApproval,
        runner: Any,
        num_reqs: int,
        num_reqs_padded: int,
        num_tokens: int,
        num_tokens_padded: int,
        cudagraph_mode: str,
    ) -> tuple[str, list[dict[str, Any]]]:
        if (
            block_table_approval.batch_id
            != slot_mapping_approval.batch_id
        ):
            raise RuntimeError(
                "block-table and slot-mapping "
                "batch IDs differ"
            )

        fixed_records = [
            self._descriptor_record(
                record=record,
                block_table_approval=(
                    block_table_approval
                ),
                slot_mapping_approval=(
                    slot_mapping_approval
                ),
                runner=runner,
            )
            for record in descriptor_records
        ]

        if not fixed_records:
            raise RuntimeError(
                "no fixed descriptor records were captured"
            )

        root = self._descriptor_root(
            batch_id=block_table_approval.batch_id,
            num_reqs=num_reqs,
            num_reqs_padded=num_reqs_padded,
            num_tokens=num_tokens,
            num_tokens_padded=num_tokens_padded,
            cudagraph_mode=cudagraph_mode,
            records=fixed_records,
        )

        return root, fixed_records

    def on_descriptor_ready(
        self,
        *,
        block_table_approval: BlockTableApproval | None,
        slot_mapping_approval: SlotMappingApproval | None,
        runner: Any,
        descriptor_records: list[dict[str, Any]],
        num_reqs: int,
        num_reqs_padded: int,
        num_tokens: int,
        num_tokens_padded: int,
        cudagraph_mode: str,
    ) -> DescriptorApproval | None:
        if not self.enabled:
            return None

        if block_table_approval is None:
            self._fail(
                "descriptor generated without "
                "block-table approval"
            )
            return None

        if slot_mapping_approval is None:
            self._fail(
                "descriptor generated without "
                "slot-mapping approval"
            )
            return None

        root, fixed_records = self._fixed_descriptor(
            descriptor_records=descriptor_records,
            block_table_approval=(
                block_table_approval
            ),
            slot_mapping_approval=(
                slot_mapping_approval
            ),
            runner=runner,
            num_reqs=num_reqs,
            num_reqs_padded=num_reqs_padded,
            num_tokens=num_tokens,
            num_tokens_padded=num_tokens_padded,
            cudagraph_mode=cudagraph_mode,
        )

        assert self.backend is not None

        try:
            (
                backend_approval,
                use_ticket,
            ) = self.backend.approve_descriptor(
                block_table_approval.backend_approval,
                slot_mapping_approval.backend_approval,
                descriptor_root=bytes.fromhex(root),
                num_reqs=num_reqs,
                num_reqs_padded=num_reqs_padded,
                num_tokens=num_tokens,
                num_tokens_padded=num_tokens_padded,
                cudagraph_mode=cudagraph_mode,
                record_count=len(fixed_records),
            )
        except BackendTZKVError as exc:
            self._fail(
                "backend rejected descriptor approval",
                error=str(exc),
            )
            return None

        approval = DescriptorApproval(
            schema_version=DESCRIPTOR_SCHEMA_VERSION,
            batch_id=block_table_approval.batch_id,
            descriptor_root=root,
            num_reqs=num_reqs,
            num_reqs_padded=num_reqs_padded,
            num_tokens=num_tokens,
            num_tokens_padded=num_tokens_padded,
            cudagraph_mode=cudagraph_mode,
            record_count=len(fixed_records),
            backend_approval=backend_approval,
            use_ticket=use_ticket,
        )

        self._event(
            "descriptor_ready",
            schema_version=(
                DESCRIPTOR_SCHEMA_VERSION
            ),
            batch_id=approval.batch_id,
            descriptor_root=root,
            num_reqs=num_reqs,
            num_reqs_padded=num_reqs_padded,
            num_tokens=num_tokens,
            num_tokens_padded=num_tokens_padded,
            cudagraph_mode=cudagraph_mode,
            records=fixed_records,
        )

        return approval

    def on_descriptor_use(
        self,
        *,
        approval: DescriptorApproval | None,
        block_table_approval: BlockTableApproval | None,
        slot_mapping_approval: SlotMappingApproval | None,
        runner: Any,
        descriptor_records: list[dict[str, Any]],
        num_reqs: int,
        num_reqs_padded: int,
        num_tokens: int,
        num_tokens_padded: int,
        cudagraph_mode: str,
    ) -> bool:
        if not self.enabled:
            return True

        if approval is None:
            return self._fail(
                "missing descriptor approval"
            )

        if (
            block_table_approval is None
            or slot_mapping_approval is None
        ):
            return self._fail(
                "missing block-table or slot-mapping "
                "approval at descriptor use"
            )

        root, fixed_records = self._fixed_descriptor(
            descriptor_records=descriptor_records,
            block_table_approval=(
                block_table_approval
            ),
            slot_mapping_approval=(
                slot_mapping_approval
            ),
            runner=runner,
            num_reqs=num_reqs,
            num_reqs_padded=num_reqs_padded,
            num_tokens=num_tokens,
            num_tokens_padded=num_tokens_padded,
            cudagraph_mode=cudagraph_mode,
        )

        if approval.schema_version != (
            DESCRIPTOR_SCHEMA_VERSION
        ):
            return self._fail(
                "descriptor schema version changed"
            )

        if approval.batch_id != (
            block_table_approval.batch_id
        ):
            return self._fail(
                "descriptor and block-table "
                "batch IDs differ"
            )

        if approval.batch_id != (
            slot_mapping_approval.batch_id
        ):
            return self._fail(
                "descriptor and slot-mapping "
                "batch IDs differ"
            )

        if root != approval.descriptor_root:
            return self._fail(
                "fixed descriptor changed before use",
                expected=approval.descriptor_root,
                actual=root,
            )

        if len(fixed_records) != approval.record_count:
            return self._fail(
                "descriptor record count changed "
                "before use"
            )

        assert self.backend is not None

        try:
            self.backend.verify_descriptor_use(
                approval.use_ticket,
                bt_group_roots=tuple(
                    bytes.fromhex(value)
                    for value in block_table_approval.group_roots
                ),
                slot_mapping_roots=tuple(
                    bytes.fromhex(value)
                    for value in slot_mapping_approval.slot_mapping_roots
                ),
                descriptor_root=bytes.fromhex(root),
                num_reqs=num_reqs,
                num_reqs_padded=num_reqs_padded,
                num_tokens=num_tokens,
                num_tokens_padded=num_tokens_padded,
                cudagraph_mode=cudagraph_mode,
                record_count=len(fixed_records),
            )
        except BackendTZKVError as exc:
            return self._fail(
                "backend rejected descriptor final use",
                error=str(exc),
            )

        self._event(
            "descriptor_use",
            schema_version=(
                DESCRIPTOR_SCHEMA_VERSION
            ),
            batch_id=approval.batch_id,
            descriptor_root=root,
            record_count=len(fixed_records),
        )

        return True


_HOOKS: TZKVRuntimeHooks | None = None


def get_tzkv_runtime_hooks() -> TZKVRuntimeHooks:
    global _HOOKS

    if _HOOKS is None:
        _HOOKS = TZKVRuntimeHooks()

    return _HOOKS
