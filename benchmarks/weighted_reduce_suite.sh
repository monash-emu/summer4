#!/bin/sh
# Every form x solver of benchmarks/weighted_reduce.py, one line each. Extra args pass through.
set -e
for solver in euler tsit5; do
  for form in terms foi plain; do
    python benchmarks/weighted_reduce.py --form "$form" --solver "$solver" "$@"
  done
done
