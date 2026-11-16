import argparse


def main(argv=None):
    p = argparse.ArgumentParser(prog="dejastream")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bench", help="offline benchmark of all strategies")
    b.add_argument("--seed", type=int, default=7)
    b.add_argument("--seg-len", type=int, default=1500)
    s = sub.add_parser("serve", help="live pipeline + dashboard")
    s.add_argument("--rate", type=int, default=600, help="samples per second")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--seed", type=int, default=7)
    s.add_argument("--loops", type=int, default=0, help="passes over the schedule (0 = forever)")
    a = p.parse_args(argv)
    if a.cmd == "bench":
        from .bench import run
        run(seed=a.seed, seg_len=a.seg_len)
    else:
        from .server import serve
        serve(rate=a.rate, port=a.port, seed=a.seed, loops=a.loops)
    return 0
