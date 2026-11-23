"""Run the existing suite (python -m unittest discover -s tests) in-process and measure STATEMENT coverage with sys.monitoring."""
import ast, os, sys, unittest, time, collections
sys.path.insert(0, os.getcwd())
ROOT = os.path.abspath("datastream") + os.sep
mon = sys.monitoring; TOOL = mon.COVERAGE_ID
hits = collections.defaultdict(set)
def on_line(code, line):
    fn = code.co_filename
    if fn.startswith(ROOT): hits[fn].add(line)
    return mon.DISABLE
mon.use_tool_id(TOOL, "audit")
mon.register_callback(TOOL, mon.events.LINE, on_line)
mon.set_events(TOOL, mon.events.LINE)
t0 = time.time()
suite = unittest.defaultTestLoader.discover("tests")
res = unittest.TextTestRunner(verbosity=0, stream=open(os.devnull, "w")).run(suite)
mon.set_events(TOOL, 0)
print(f"tests run={res.testsRun} failures={len(res.failures)} errors={len(res.errors)} skipped={len(res.skipped)} seconds={time.time()-t0:.1f}")
rows = []
for f in sorted(os.listdir("datastream")):
    if not f.endswith(".py"): continue
    path = os.path.abspath(f"datastream/{f}")
    tree = ast.parse(open(path).read())
    stmts = {n.lineno for n in ast.walk(tree) if isinstance(n, ast.stmt) and not isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)) and not (isinstance(n, ast.Expr) and isinstance(getattr(n, "value", None), ast.Constant))}
    cov = stmts & hits[path]
    rows.append((f, len(stmts), len(cov)))
print(f"{'module':16} {'stmts':>6} {'hit':>5} {'cover':>7}")
tot = totc = 0
for f, n, c in rows:
    print(f"{f:16} {n:6d} {c:5d} {100*c/max(n,1):6.1f}%"); tot += n; totc += c
print(f"{'TOTAL':16} {tot:6d} {totc:5d} {100*totc/tot:6.1f}%")
print("\nUNCOVERED STATEMENT LINES (modules in the main path):")
for f in ("mv_engine.py", "matcher.py", "pipeline.py", "store.py", "server.py", "bus.py", "engine.py", "detector.py"):
    path = os.path.abspath(f"datastream/{f}"); tree = ast.parse(open(path).read())
    stmts = sorted({n.lineno for n in ast.walk(tree) if isinstance(n, ast.stmt) and not isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)) and not (isinstance(n, ast.Expr) and isinstance(getattr(n, "value", None), ast.Constant))})
    miss = [l for l in stmts if l not in hits[path]]
    # compress to ranges
    rng, start = [], None
    for l in miss:
        if start is None: start = prev = l
        elif l == prev + 1 or l == prev + 2: prev = l
        else: rng.append((start, prev)); start = prev = l
    if start is not None: rng.append((start, prev))
    print(f"  {f}: " + ", ".join(f"{a}-{b}" if a != b else str(a) for a, b in rng))
