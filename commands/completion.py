"""Interactive CLI tab-completion helpers."""

from __future__ import annotations

from typing import Iterable


def command_name_completions(
    *,
    text: str,
    line_buffer: str,
    command_names: Iterable[str],
    macro_names: Iterable[str] = (),
) -> list[str]:
    """Return completion candidates for the current command name.

    This completion is intentionally conservative: it only completes the *first*
    token of the current command segment. For compound command lines we treat
    `;` as a segment separator and complete based on the last segment.
    """
    buf = line_buffer or ""
    # Only consider the active segment for compound commands.
    segment = buf.split(";")[-1].lstrip()
    if not segment:
        prefix = ""
    else:
        # Only complete the first token (command name).
        if " " in segment:
            return []
        prefix = segment

    # Readline passes `text` as the current word, but for safety we filter on
    # both `text` and the current segment prefix.
    want = (text or "").strip()
    if want:
        starts = want
    else:
        starts = prefix

    names = set(str(n) for n in command_names) | set(str(n) for n in macro_names)
    matches = sorted(n for n in names if n.startswith(starts))
    return matches
