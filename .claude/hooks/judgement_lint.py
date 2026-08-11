#!/usr/bin/env python3
"""PostToolUse lint: keep the JUDGEMENT (unmeasured) tag from becoming decoration.

`claude/guide/01-the-data.md` §1 sets two rules for the tag, and both are mechanical:

  1. Every tag names its falsifier. A judgement with no stated falsifier is either
     measurable now (measure it) or not a claim (delete it).
  2. A claim with a number, a command, or a file citation in it is NOT tagged. The whole
     value of the tag is that it separates measured from unmeasured; tagging a measurement
     destroys that separation.

Rule 1 has no false positives and runs by default. Rule 2 is a heuristic -- a judgement may
legitimately quote a number it is *not* resting on, and all three tags in this repository
that trip it are fine -- so it runs only under `--strict`, as a periodic sweep rather than
on every edit. A lint that cries wolf on every save is the same failure as a tag on every
sentence.

Neither blocks. Exit 2 routes stderr back to the agent as feedback; the edit already
happened.

Usage: reads the PostToolUse payload on stdin. Also runnable directly on paths:
    python3 .claude/hooks/judgement_lint.py claude/DESI234/README.md
    python3 .claude/hooks/judgement_lint.py --all [--strict]
"""
import glob
import json
import os
import re
import sys

TAG = "JUDGEMENT (unmeasured)"
# No colon required: the README form is "*Falsified by:* ..." and the config.py comment form
# is "Falsified by a measured rate ...". Both name a falsifier, which is the rule.
FALSIFIER = re.compile(r"falsified by", re.I)
# A bare integer or decimal, with or without a unit. Deliberately loose.
NUMBER = re.compile(r"(?<![\w.])\d+(\.\d+)?\s*(\"|'|arcsec|px|pixel|sigma|sd|%)?(?![\w])")
CITATION = re.compile(r"`[^`]*(\.py|\.md|\.npz|\.json|\.fits|/)[^`]*`")

# Files the tag convention applies to. The guide itself is excluded: it *documents* the
# convention, so its worked example is prose about a tag rather than a tag. The exclusion is
# by directory, not by filename, since the guide is now `claude/guide/*.md` -- a basename
# check would have gone quietly stale at the split and started linting the guide's own
# worked example.
PATTERNS = ("claude/**/*.md", "claude/**/config.py")
EXCLUDE_DIRS = ("claude/guide",)


def blocks(text):
    """Yield (line_no, paragraph_text) for each paragraph containing the tag."""
    lines = text.splitlines()
    para, start = [], 0
    for i, line in enumerate(lines, 1):
        if line.strip():
            if not para:
                start = i
            para.append(line)
        else:
            if any(TAG in ln for ln in para):
                yield start, "\n".join(para)
            para = []
    if para and any(TAG in ln for ln in para):
        yield start, "\n".join(para)


def lint(path, strict=False):
    norm = os.path.normpath(path).replace(os.sep, "/")
    if any(f"/{d}/" in f"/{norm}/" for d in EXCLUDE_DIRS):
        return []
    try:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
    except (OSError, UnicodeDecodeError):
        return []
    out = []
    for line_no, para in blocks(text):
        if not FALSIFIER.search(para):
            out.append(f"{path}:{line_no}  tag with no falsifier -- "
                       f"add '*Falsified by:* <run or measurement>', or delete the claim")
        if not strict:
            continue
        # Strip the falsifier clause before looking for evidence: a falsifier legitimately
        # names a file or a target number.
        claim = FALSIFIER.split(para)[0]
        if CITATION.search(claim):
            out.append(f"{path}:{line_no}  tagged claim cites a file -- "
                       f"if it is backed by that file it is a measurement, not a judgement")
        elif NUMBER.search(claim):
            hit = NUMBER.search(claim).group(0).strip()
            out.append(f"{path}:{line_no}  tagged claim contains a number ({hit!r}) -- "
                       f"check it is context, not the claim's evidence")
    return out


def targets_from_stdin():
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return []
    path = (payload.get("tool_input") or {}).get("file_path", "")
    if not path:
        return []
    rel = os.path.relpath(path, payload.get("cwd") or os.getcwd())
    if rel.startswith("claude" + os.sep) and (rel.endswith(".md") or rel.endswith("config.py")):
        return [path]
    return []


def main():
    argv = sys.argv[1:]
    strict = "--strict" in argv
    argv = [a for a in argv if a != "--strict"]
    if argv == ["--all"]:
        paths = [p for pat in PATTERNS for p in glob.glob(pat, recursive=True)]
    elif argv:
        paths = argv
    else:
        paths = targets_from_stdin()

    problems = [msg for p in paths for msg in lint(p, strict)]
    if not problems:
        return 0
    print("JUDGEMENT tag lint (guide section 1):", file=sys.stderr)
    for msg in problems:
        print("  " + msg, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
