from __future__ import annotations

import re

from .models import FrameMetadata


def frame_modality(frame: FrameMetadata) -> str:
    """Return the normalized camera modality without guessing from substrings.

    The official response may provide an explicit modality. Older/mock payloads can
    omit it, so video-name tokens are used as a conservative compatibility fallback.
    """

    value = str(getattr(frame, "modality", "unknown")).casefold()
    if value in {"rgb", "thermal"}:
        return value
    tokens = set(re.findall(r"[a-z0-9]+", frame.video_name.casefold()))
    if tokens.intersection({"thermal", "termal", "infrared", "ir"}):
        return "thermal"
    return "rgb" if "rgb" in tokens else "unknown"
