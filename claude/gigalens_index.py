"""Generate a symbol index of the installed gigalens package, for fast lookup.

Run this, then read ``claude/GIGALENS_INDEX.md``, before writing anything that might
already exist in the library. gigalens has ~90 modules; grepping it from scratch every
time is slow, and the failure mode is not "wasted effort" but a hand-rolled
reimplementation that is subtly wrong in a way the library's own tests would have caught.

Two design decisions worth stating:

- **It parses with ``ast``, never imports.** Importing gigalens costs a JAX initialisation
  and pulls in GPU state; parsing is instant and cannot execute module-level code. The
  cost is that dynamically-constructed names (there are few) do not appear.
- **The package is located through ``importlib``, not a hardcoded path.** gigalens is
  installed editable from a directory that is a *different project* (see CLAUDE.md: do
  not edit it from here). If that install ever moves, this keeps working and the index
  records where it actually pointed.

The output is a snapshot and carries its generation date and the resolved source path in
its header, so a stale index is visible as stale rather than quietly wrong. Regenerate it
after pulling nanolens.
"""

import argparse
import ast
import datetime as _dt
import importlib.util
from pathlib import Path

# Subtrees deliberately omitted from the index, with the reason. `old_api` is the
# superseded PhysicalModel/ForwardProbModel/ModellingSequence interface -- indexing it
# would surface names that look usable and are not (CLAUDE.md: "If you find yourself
# importing from it, you are working from a stale example"). The checkpoint dirs are
# Jupyter's autosaves, i.e. duplicate stale copies of files indexed elsewhere.
SKIP_PARTS = {"old_api", ".ipynb_checkpoints", "__pycache__"}

# Cap on a summary line. Long enough for a real one-line docstring, short enough that the
# whole index stays scannable.
SUMMARY_CHARS = 110


def first_line(node):
    """First sentence-ish line of a node's docstring, squashed to one line."""
    doc = ast.get_docstring(node)
    if not doc:
        return ""
    line = " ".join(doc.strip().split("\n")[0].split())
    return line[: SUMMARY_CHARS - 1] + "…" if len(line) > SUMMARY_CHARS else line


def signature(node):
    """Render a def/class parameter list, without annotations or defaults.

    Annotations are dropped on purpose: they roughly double the width and the index is for
    *finding* a symbol, not for calling it correctly. Read the source for that -- the
    index prints the file:line so it is one jump away.
    """
    if isinstance(node, ast.ClassDef):
        bases = [ast.unparse(b) for b in node.bases]
        return f"({', '.join(bases)})" if bases else ""
    a = node.args
    names = [p.arg for p in a.posonlyargs + a.args]
    if a.vararg:
        names.append("*" + a.vararg.arg)
    elif a.kwonlyargs:
        names.append("*")
    names += [p.arg for p in a.kwonlyargs]
    if a.kwarg:
        names.append("**" + a.kwarg.arg)
    return "(" + ", ".join(n for n in names if n != "self") + ")"


def public(name):
    return not name.startswith("_") or name == "__init__"


def index_module(path, root):
    """Return ``(rel_path, module_summary, [entry, ...])`` for one source file.

    An entry is ``(kind, name, signature, summary, lineno, methods)``.
    """
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError as exc:  # a broken file in the dependency is worth surfacing
        return str(path.relative_to(root)), f"!! SyntaxError: {exc}", []

    entries = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and public(node.name):
            entries.append(("def", node.name, signature(node), first_line(node),
                            node.lineno, []))
        elif isinstance(node, ast.ClassDef) and public(node.name):
            methods = [
                (m.name, signature(m), first_line(m), m.lineno)
                for m in node.body
                if isinstance(m, (ast.FunctionDef, ast.AsyncFunctionDef)) and public(m.name)
            ]
            entries.append(("class", node.name, signature(node), first_line(node),
                            node.lineno, methods))
    return str(path.relative_to(root)), first_line(tree), entries


def gigalens_root():
    """Resolve the installed package directory, or raise with an actionable message."""
    spec = importlib.util.find_spec("gigalens")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit(
            "gigalens is not importable. Activate the gigajax7.0 env, or run this with "
            "/global/homes/s/seanjx/.conda/envs/gigajax7.0/bin/python.")
    return Path(list(spec.submodule_search_locations)[0])


def _docs_section(root):
    """List the sibling prose docs, if the install is a source checkout.

    An editable install points at ``<repo>/src/gigalens``, so the project's own
    documentation sits two levels up. It answers a different class of question than a
    symbol index does -- known issues and worked examples rather than "what is this
    called" -- and it is the thing to read *before* blaming your own code.
    """
    docs = root.parent.parent / "docs"
    if not docs.is_dir():
        return []
    lines = ["## Prose docs (read these before blaming your own code)", ""]
    for p in sorted(docs.rglob("*.md")) + sorted(docs.rglob("*.ipynb")):
        if "_build" in p.parts or ".ipynb_checkpoints" in p.parts:
            continue
        lines.append(f"- `{p}`")
    lines.append("")
    return lines


def render(root, modules):
    out = [
        "# gigalens symbol index",
        "",
        f"Generated by `claude/gigalens_index.py` on "
        f"{_dt.date.today().isoformat()} from `{root}`.",
        "",
        "**Regenerate after pulling nanolens** — this is a snapshot, and gigalens is a",
        "separate, actively developed project that must not be edited from this repo.",
        "",
        "```",
        "/global/homes/s/seanjx/.conda/envs/gigajax7.0/bin/python claude/gigalens_index.py",
        "```",
        "",
        "`old_api/` is excluded on purpose: it is the superseded "
        "`PhysicalModel`/`ForwardProbModel`/`ModellingSequence` interface, and importing "
        "from it means you are working from a stale example.",
        "",
    ]

    out += _docs_section(root)

    # Module table first, so the head of the file answers "which module is this in?"
    # without reading the (large) per-symbol detail below. Grep the detail for a name.
    out += ["## Modules at a glance", ""]
    for rel, summary, entries in modules:
        if not entries and not summary:
            continue
        anchor = "#" + rel.replace("/", "").replace(".", "").replace("_", "-")
        out.append(f"- [`{rel}`]({anchor}) — {summary or '(no module docstring)'}")
    out.append("")

    for rel, summary, entries in modules:
        if not entries and not summary:
            continue
        out.append(f"## `{rel}`")
        if summary:
            out.append(f"{summary}")
        out.append("")
        for kind, name, sig, doc, lineno, methods in entries:
            tag = "class" if kind == "class" else "def"
            out.append(f"- **`{tag} {name}{sig}`** — `{rel}:{lineno}`"
                       + (f"  \n  {doc}" if doc else ""))
            for mname, msig, mdoc, mline in methods:
                out.append(f"    - `.{mname}{msig}` — `:{mline}`"
                           + (f" — {mdoc}" if mdoc else ""))
        out.append("")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="claude/GIGALENS_INDEX.md",
                    help="Output markdown path (default: claude/GIGALENS_INDEX.md)")
    args = ap.parse_args()

    root = gigalens_root()
    paths = sorted(
        p for p in root.rglob("*.py")
        if not SKIP_PARTS & set(p.parts) and p.name != "__init__.py"
    )
    modules = [index_module(p, root) for p in paths]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render(root, modules))

    n_sym = sum(len(e) + sum(len(x[5]) for x in e) for _, _, e in modules)
    print(f"Indexed {len(modules)} modules, {n_sym} symbols from {root}")
    print(f"Wrote {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
