import argparse


def main(argv=None):
    p = argparse.ArgumentParser(prog="datastream")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bench", help="offline benchmark of all strategies")
    b.add_argument("--seed", type=int, default=7)
    b.add_argument("--seg-len", type=int, default=1500)
    m = sub.add_parser("bench-mv", help="multi-seed, multi-scenario benchmark incl. retrieval metrics")
    m.add_argument("--seeds", type=int, default=5)
    m.add_argument("--scenarios", nargs="*", help="subset of scenario names")
    s = sub.add_parser("serve", help="live pipeline + dashboard")
    s.add_argument("--rate", type=int, default=600, help="samples per second")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--loops", type=int, default=0, help="passes over the schedule or file (0 = forever)")
    s.add_argument("--arff", help="stream a real ARFF file instead of the synthetic concepts")
    s.add_argument("--target", help="target column of the ARFF file (nominal)")
    s.add_argument("--limit", type=int, help="use only the first N rows of the file")
    s.add_argument("--features", help="comma separated feature columns (default: all but high-cardinality nominal columns)")
    a = p.parse_args(argv)
    if a.cmd == "bench":
        from .bench import run
        run(seed=a.seed, seg_len=a.seg_len)
    elif a.cmd == "bench-mv":
        from .bench_mv import run, to_markdown
        r = run(seeds=tuple(range(1, a.seeds + 1)), scenarios=a.scenarios)
        print(to_markdown(r))
    else:
        from .server import serve
        serve(rate=a.rate, port=a.port, seed=a.seed, loops=a.loops, arff=a.arff, target=a.target, limit=a.limit,
              features=a.features.split(",") if a.features else None)
    return 0
