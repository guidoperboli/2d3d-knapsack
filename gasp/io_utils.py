"""Robust checkpoint writing for long campaigns.

Campaigns rewrite their JSON checkpoint after every instance -- hundreds
of times. When the results directory lives inside a syncing folder
(Dropbox, OneDrive, Google Drive) on Windows, the sync client may hold
the file open exactly when we try to rewrite it, and the write fails
with PermissionError (errno 13) or a transient OSError. A single failed
write then crashes a multi-hour run.

`write_json_atomic` makes the write robust in two ways:
  * atomic: it writes to a temporary file in the same directory and then
    os.replace()s it onto the target, so a reader (or sync client) never
    sees a half-written file, and the rename is atomic on Windows and
    POSIX alike;
  * retried: if the replace is briefly blocked by a lock, it retries a
    few times with a short backoff before giving up.

This is a no-op behaviourally (same file contents); it only changes how
the bytes reach disk.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path


def write_json_atomic(path, obj, *, indent=None, retries=8,
                      backoff=0.25):
    """Write `obj` as JSON to `path` atomically, retrying on transient
    lock errors (common inside Dropbox/OneDrive on Windows)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(obj, indent=indent)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    last_err = None
    for attempt in range(retries):
        try:
            tmp.write_text(data, encoding="utf-8")
            os.replace(tmp, path)        # atomic on Win + POSIX
            return
        except (PermissionError, OSError) as e:
            last_err = e
            # the sync client is probably holding the file; wait and retry
            time.sleep(backoff * (attempt + 1))
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass
    # all retries exhausted: surface a clear, actionable message
    raise RuntimeError(
        f"Impossibile scrivere il checkpoint {path} dopo {retries} "
        f"tentativi (ultimo errore: {last_err}). Probabile lock di "
        f"Dropbox/OneDrive: spostare la cartella results/ fuori dalla "
        f"cartella sincronizzata, oppure mettere in pausa la sync "
        f"durante il calcolo."
    ) from last_err
