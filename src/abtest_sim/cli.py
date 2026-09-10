"""Команды проекта: python -m abtest_sim.cli {data,sim,figures,benchmark}."""

import argparse
import time


def cmd_data(args):
    from .data import download

    download(force=args.force)


def cmd_sim(args):
    from .scenarios import SCENARIOS, run_scenario

    for name in args.scenario or SCENARIOS:
        start = time.perf_counter()
        df = run_scenario(name, n_jobs=args.jobs)
        print(f"{name:<12} {len(df):>10,} строк {time.perf_counter() - start:7.1f} с")


def cmd_figures(args):
    from .plots import make_all

    for path in make_all():
        print(path)


def cmd_benchmark(args):
    from .benchmark import run_benchmark

    print(run_benchmark(args.jobs).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(prog="abtest_sim")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("data", help="скачать датасет Cookie Cats в data/")
    p.add_argument("--force", action="store_true", help="скачать заново")
    p.set_defaults(func=cmd_data)

    p = sub.add_parser("sim", help="посчитать симуляции в results/")
    p.add_argument("-s", "--scenario", action="append", help="имя сценария, можно несколько")
    p.add_argument("-j", "--jobs", type=int, help="число процессов")
    p.set_defaults(func=cmd_sim)

    p = sub.add_parser("figures", help="построить графики в figures/")
    p.set_defaults(func=cmd_figures)

    p = sub.add_parser("benchmark", help="сравнить цикл, векторизацию и процессы")
    p.add_argument("-j", "--jobs", type=int, default=4, help="число процессов")
    p.set_defaults(func=cmd_benchmark)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
