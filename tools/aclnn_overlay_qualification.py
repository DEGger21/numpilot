#!/usr/bin/env python3
"""Thin CLI for the official ACLNN overlay qualification producer.

The producer itself lives in :mod:`numpilot.gaps.official_producer` so the
repository tests can exercise the same contract without a subprocess.  This
wrapper only makes the local checkout importable when the tool is run directly
from ``tools/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from numpilot.gaps.official_producer import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
