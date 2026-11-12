"""Idempotently wrap every tabular in the generated table files in \\resizebox so wide tables fit the text width."""
import sys, re, pathlib
for p in sys.argv[1:]:
    t = pathlib.Path(p).read_text()
    if "resizebox" in t:
        print("skip", p); continue
    t = re.sub(r"\\begin\{tabular\}", r"\\resizebox{\\ifdim\\width>\\linewidth\\linewidth\\else\\width\\fi}{!}{\\begin{tabular}", t)
    t = t.replace(r"\end{tabular}", r"\end{tabular}}")
    pathlib.Path(p).write_text(t); print("wrapped", p)
