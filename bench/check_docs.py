"""Check local markdown links resolve and key numbers agree across docs."""
import re, pathlib

root = pathlib.Path(".")
files = [root / "README.md"] + sorted(root.glob("docs/*.md")) + [root / "COMPARISON.md"]
bad = []
for f in files:
    if not f.exists():
        continue
    text = f.read_text()
    for m in re.finditer(r"\[([^\]]+)\]\(([^)#][^)]*)\)", text):
        target = m.group(2).split("#")[0].strip().strip("<> ")
        if not target or target.startswith("http") or target.startswith("mailto"):
            continue
        p = (f.parent / target).resolve()
        if not p.exists():
            bad.append((str(f), target))
print("broken links:", len(bad))
for f, t in bad[:12]:
    print("  ", f, "->", t)

# key numbers that must agree wherever they appear
checks = {
    "0.5B decode": ["870.9", "871"],
    "0.5B prefill": ["48,429", "48.4k", "48,533"],
}
print("\nnumbers of interest:")
for name, pats in checks.items():
    hits = {}
    # only the current-state docs must agree; evidence/scorecard are a chronological
    # log and legitimately contain the numbers as measured at the time
    current = [root / "README.md", root / "docs/SUMMARY.md", root / "docs/final_table.md"]
    for f in current:
        if not f.exists():
            continue
        t = f.read_text()
        found = [p for p in pats if p in t]
        if found:
            hits[str(f)] = found
    print(" ", name, hits)
