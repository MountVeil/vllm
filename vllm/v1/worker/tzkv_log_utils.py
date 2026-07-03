# SPDX-License-Identifier: Apache-2.0
#
# Shared TZKV logging switch.
#
# TZKV_LOG_MODE=jsonl  -> write JSONL logs
# TZKV_LOG_MODE=off    -> disable per-event JSONL logs
#
# Backward-compatible:
# TZKV_LOG_ENABLE=0 also disables logs.

from __future__ import annotations

import os

_OFF_VALUES = {"0", "false", "no", "off", "none", "disabled"}


def tzkv_log_enabled() -> bool:
    mode = os.getenv("TZKV_LOG_MODE", os.getenv("TZKV_LOG_ENABLE", "jsonl"))
    return str(mode).strip().lower() not in _OFF_VALUES
