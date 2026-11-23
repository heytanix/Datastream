"""Which functions execute in the DEFAULT live pipeline (Pipeline(), default engines, synthetic stream) for ~10 s?"""
import ast, os, sys, threading, time, tempfile, collections
sys.path.insert(0, os.getcwd())
executed = collections.Counter()
ROOT = os.path.abspath("dejastream") + os.sep
def prof(frame, event, arg):
    if event == "call":
        fn = frame.f_code.co_filename
        if fn.startswith(ROOT) and "/research/" not in fn:
            executed[(os.path.relpath(fn), frame.f_code.co_name, frame.f_code.co_firstlineno)] += 1
threading.setprofile(prof); sys.setprofile(prof)
from dejastream.pipeline import Pipeline
d = tempfile.mkdtemp()
p = Pipeline(rate=2500, seed=7, loops=1, db_path=os.path.join(d, "audit.db"))   # default engines; temp DB (project DB untouched)
p.start(); time.sleep(10); snap = p.snapshot(); p.close()
sys.setprofile(None); threading.setprofile(None)
print("samples processed:", snap["t"], "| strategies:", list(snap["summary"]))
defs = []
for f in sorted(os.listdir("dejastream")):
    if f.endswith(".py"):
        t = ast.parse(open(f"dejastream/{f}").read())
        for n in ast.walk(t):
            if isinstance(n, ast.FunctionDef): defs.append((f"dejastream/{f}", n.name, n.lineno))
ran = {(a, b) for (a, b, c) in executed}
print("\nDEFINED but NOT executed by the default pipeline (module-level + methods; research/ excluded):")
byfile = collections.defaultdict(list)
for f, name, ln in defs:
    if (f, name) not in ran: byfile[f].append(f"{name}@{ln}")
for f, names in byfile.items():
    print(f"  {f}: {', '.join(names)}")
