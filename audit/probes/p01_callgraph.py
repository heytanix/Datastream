"""Static reachability: which definitions are referenced by the production path (cli -> server -> pipeline -> strategies -> engines)?
Name-based and approximate: a name counts as 'referenced' if it appears as an identifier/attribute in a reachable module."""
import ast, os, sys, collections
ROOT = "datastream"
mods = {f[:-3]: os.path.join(ROOT, f) for f in os.listdir(ROOT) if f.endswith(".py")}
def imports(path):
    t = ast.parse(open(path).read()); out = set()
    for n in ast.walk(t):
        if isinstance(n, ast.ImportFrom) and n.level == 1 and n.module: out.add(n.module.split(".")[0])
        if isinstance(n, ast.ImportFrom) and n.level == 1 and not n.module:
            out |= {a.name for a in n.names}
        if isinstance(n, ast.Import):
            pass
    return {m for m in out if m in mods}
graph = {m: imports(p) for m, p in mods.items()}
# lazy (function-level) imports are included by ast.walk, which is what we want
seen, todo = set(), ["cli"]
while todo:
    m = todo.pop()
    if m in seen: continue
    seen.add(m); todo += list(graph[m])
print("MODULES REACHABLE FROM cli.main:", sorted(seen))
print("MODULES NOT REACHABLE FROM cli.main:", sorted(set(mods) - seen - {"__init__"}))
# which of the entry commands reach which modules
for entry in ("server", "bench", "bench_mv", "pipeline"):
    s2, t2 = set(), [entry]
    while t2:
        m = t2.pop()
        if m in s2: continue
        s2.add(m); t2 += list(graph[m])
    print(f"  via {entry}: {sorted(s2)}")
names_used = collections.defaultdict(set)
for m in seen:
    t = ast.parse(open(mods[m]).read())
    for n in ast.walk(t):
        if isinstance(n, ast.Name): names_used[n.id].add(m)
        if isinstance(n, ast.Attribute): names_used[n.attr].add(m)
tests = "".join(open(os.path.join("tests", f)).read() for f in os.listdir("tests") if f.endswith(".py"))
print("\nDEFINITIONS in reachable modules never referenced by another reachable module (excluding own module), and whether tests reference them:")
for m in sorted(seen):
    t = ast.parse(open(mods[m]).read())
    for n in ast.walk(t):
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and not n.name.startswith("_"):
            users = names_used.get(n.name, set()) - {m}
            if not users:
                print(f"  {m}.py:{n.lineno:<4} {n.name:32} tests-reference={'yes' if n.name in tests else 'NO'}")
