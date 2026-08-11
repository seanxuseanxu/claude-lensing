#!/usr/bin/env python3
"""PreToolUse guard: refuse to edit the nanolens working copy from this repo.

CLAUDE.md states that `/global/u1/s/seanjx/nanolens` is a different project and must not
be edited from here, and that breakage there is to be *reported*, not patched around. That
rule currently lives only in prose, which means it holds exactly as long as the agent
remembers it. This makes it hold deterministically.

Covers Edit / Write / NotebookEdit. It does **not** cover `Bash` -- a `sed -i` or a shell
redirect into that tree still gets through, because every substring test that would catch
those also fires on harmless reads (`pip show gigalens`, `grep -r ... nanolens/`). The
guard is a floor, not a fence.

Exit 2 blocks the call and shows stderr to the agent.
"""
import json
import os
import sys

# Both spellings of the same directory: /global/u1/... and the /global/homes/... symlink.
FORBIDDEN = (
    "/global/u1/s/seanjx/nanolens",
    "/global/homes/s/seanjx/nanolens",
)

REASON = """BLOCKED: {path}

That path is inside the nanolens working copy. CLAUDE.md: "That directory is a different
project -- do not edit it from here. If something in it is broken or wrong, report it; do
not patch around it silently either."

If gigalens genuinely needs a change, say so in the response and let the human take it to
that repository. If you were about to work around a defect there, describe the defect
instead."""


def main():
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0  # never block on a malformed payload

    path = (payload.get("tool_input") or {}).get("file_path", "")
    if not path:
        return 0
    real = os.path.realpath(os.path.expanduser(path))

    for root in FORBIDDEN:
        root_real = os.path.realpath(root)
        if real == root_real or real.startswith(root_real + os.sep):
            print(REASON.format(path=path), file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
