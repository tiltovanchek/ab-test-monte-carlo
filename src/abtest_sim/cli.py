"""Команды проекта: python -m abtest_sim.cli {sim}."""

import argparse
import time


def cmd_sim(args):
    from .scenarios import SCENARIOS, run_scenario

    for name in args.scenario or SCENARIOS:
        start = time.perf_counter()
        df = run_scenario(name, n_jobs=args.jobs)
        print(f"{name:<12} {len(df):>10,} строк {time.perf_counter() - start:7.1f} с")


def main():
    parser = argparse.ArgumentParser(prog="abtest_sim")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("sim", help="посчитать симуляции в results/")
    p.add_argument("-s", "--scenario", action="append", help="имя сценария, можно несколько")
    p.add_argument("-j", "--jobs", type=int, help="число процессов")
    p.set_defaults(func=cmd_sim)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
