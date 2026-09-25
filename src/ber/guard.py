"""Run-safety for a 16 GB machine.

The full data (~22M records) makes each heavy step use several GB. Two heavy steps
at once exhausted macOS memory twice (kernel panic), so every heavy script calls
`exclusive()` first: it takes a system-wide lock and refuses to start while another
heavy step holds it. The lock is released automatically when the process exits,
even if it crashes.
"""

import fcntl
import os
import sys

LOCK_PATH = os.environ.get("BER_LOCK", "/tmp/ber_pipeline.lock")
_handle = None


def exclusive(name):
    """Take the pipeline lock or exit immediately if another heavy step is running."""
    global _handle
    if _handle is not None:          # already holding it in this process
        return
    _handle = open(LOCK_PATH, "a+")
    try:
        fcntl.flock(_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        _handle.seek(0)
        holder = _handle.read().strip() or "another step"
        sys.exit(f"[guard] refusing to start '{name}': {holder} is running. "
                 f"Run heavy steps one at a time.")
    _handle.seek(0)
    _handle.truncate()
    _handle.write(f"{name} (pid {os.getpid()})")
    _handle.flush()
