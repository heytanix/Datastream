"""Mutation sampling on a SCRATCH COPY (the project source is never modified): does the suite notice deliberate behavioural bugs?"""
import os, shutil, subprocess, sys, tempfile, time
SRC = os.getcwd(); PY = os.path.join(SRC, ".venv/bin/python")
M = [
 ("M01 matcher: never return NEW_CONCEPT",            "dejastream/matcher.py",   "        if best.D > thr_cand:", "        if best.D > 1e9:"),
 ("M02 matcher: ignore min_samples (no INSUFFICIENT)", "dejastream/matcher.py",   "        if q.n < cfg.min_samples:", "        if q.n < -1:"),
 ("M03 matcher: ignore completeness check",           "dejastream/matcher.py",   "        if best.distances.completeness < cfg.min_completeness:", "        if best.distances.completeness < -1:"),
 ("M04 matcher: protect only the caller's concept on eviction (the real ELEC2 bug)", "dejastream/matcher.py", "        keep = {cid} | ({protect} if protect else set())", "        keep = {protect or cid}"),
 ("M05 similarity: drop weight renormalisation",       "dejastream/similarity.py", "        D = sum(cfg.weights[k] * v for k, v in avail.items()) / wtot", "        D = sum(cfg.weights[k] * v for k, v in avail.items())"),
 ("M06 engine: rollback does NOT restore the old model", "dejastream/mv_engine.py", "            self.model = r.prev_model\n", "            pass\n"),
 ("M07 engine: false-alarm path disabled",             "dejastream/mv_engine.py", "        if best and best[0] == r.prev_id:", "        if False and best and best[0] == r.prev_id:"),
 ("M08 engine: Page-Hinkley error detector disabled",  "dejastream/mv_engine.py", "            if self.detector.update(int(pp != yy)) and why is None:", "            if False and self.detector.update(int(pp != yy)) and why is None:"),
 ("M09 engine: no quarantine (learn from every sample at once)", "dejastream/mv_engine.py", "            if len(self.pending) > self.quarantine:", "            if len(self.pending) > 0:"),
 ("M10 engine: only the best candidate is shadow-validated", "dejastream/mv_engine.py", "                 if i == 0 or c.D <= self.repo.thresholds(c.concept_id, res.n)[\"thr_cand\"]]", "                 if i == 0]"),
 ("M11 engine: label released one step early (off by one)", "dejastream/mv_engine.py", "        while self._delay and t - self._delay[0][0] >= self.label_delay:", "        while self._delay and t - self._delay[0][0] >= self.label_delay - 1:"),
 ("M12 validation: Bonferroni correction removed",      "dejastream/validation.py", "    a = cfg.alpha / m if cfg.selection == \"bonferroni\" else cfg.alpha", "    a = cfg.alpha"),
 ("M13 pipeline: accuracy counter counts every sample as correct", "dejastream/pipeline.py", "                hits += int(pred == r[\"y\"])", "                hits += 1"),
 ("M14 legacy engine: no quarantine",                  "dejastream/engine.py",    "                if len(self.pending) > self.quarantine:", "                if len(self.pending) > 0:"),
]
only = sys.argv[1:] 
print(f"{'mutant':78} result")
killed = 0; n = 0
for name, rel, old, new in M:
    if only and not any(name.startswith(o) for o in only): continue
    tmp = tempfile.mkdtemp(prefix="mut_")
    for d in ("dejastream", "tests"): shutil.copytree(os.path.join(SRC, d), os.path.join(tmp, d), ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(os.path.join(SRC, "run.py"), tmp)
    p = os.path.join(tmp, rel); s = open(p).read(); assert old in s, f"pattern not found for {name}"
    open(p, "w").write(s.replace(old, new, 1))
    t0 = time.time()
    r = subprocess.run([PY, "-m", "unittest", "discover", "-s", "tests", "-f"], cwd=tmp, capture_output=True, text=True, timeout=300)
    ok = r.returncode == 0; n += 1; killed += (not ok)
    first = next((l for l in r.stderr.splitlines() if l.startswith(("FAIL:", "ERROR:"))), "")
    print(f"{name:78} {'SURVIVED (suite still green)' if ok else 'killed by ' + first[:70]}  [{time.time()-t0:.0f}s]", flush=True)
    shutil.rmtree(tmp, ignore_errors=True)
print(f"\nmutation score: {killed}/{n} killed")
