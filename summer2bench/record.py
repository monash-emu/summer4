"""Record the summer2 timing matrix. No new model structure.

One cell is model × solver × step count. Build time is construction plus
``get_runner``. Compile time is the first call. The warm median is the median
of five later calls, each ending in ``jax.block_until_ready`` on the reduced
sum. The first call is not one of the five.

The parent runs each cell in its own process so an out-of-memory kill is a
``status: failed`` record rather than a lost matrix. If a cell fails, the
rest of that model is still run, then later models are recorded as not run.
"""

from __future__ import annotations

import argparse
import datetime
import importlib.metadata
import json
import os
import platform
import resource
import signal
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

import jax

jax.config.update("jax_enable_x64", True)

from models import (  # noqa: E402
    MODEL_NAMES,
    STRATIFIED_NAMES,
    load_spec,
    prepare_model,
    runner_for,
)
from time_run import reduce_runner_outputs, time_runner  # noqa: E402

RECORDED_PATH = Path(__file__).resolve().parent / "recorded.json"
SOLVERS = ("euler", "rk4")
N_WARM = 5
RECORD_KEYS = (
    "model",
    "solver",
    "steps",
    "compartments",
    "build_s",
    "compile_s",
    "warm_median_s",
    "warm_s",
    "jax_version",
    "summerepi2_version",
    "dtype",
    "machine",
    "date",
    "status",
    "reason",
    "peak_rss_bytes",
)


def machine_description() -> str:
    """Host identity copied onto every record. Same string for the whole matrix."""
    parts = [platform.platform(), platform.machine()]
    cpu = _sysctl("machdep.cpu.brand_string")
    if cpu:
        parts.append(cpu)
    mem = _sysctl("hw.memsize")
    if mem and mem.isdigit():
        gib = int(mem) / (1024**3)
        parts.append(f"{gib:.0f} GiB")
    return "; ".join(parts)


def cell_timeout_s() -> float:
    """Per-cell wall limit. A cell that does not finish is ``status: failed``."""
    raw = os.environ.get("SUMMER2_CELL_TIMEOUT_S", "7200")
    return float(raw)


def model_order() -> tuple[str, ...]:
    """Ladder order. A failure stops after the model that failed, not before."""
    return MODEL_NAMES + STRATIFIED_NAMES


def execute_cell(model: str, solver: str, steps: int, machine: str, date: str) -> dict[str, Any]:
    """Time one cell inside this process. Exceptions become a failed record."""
    spec = load_spec()
    if model not in model_order():
        raise KeyError(model)
    if solver not in SOLVERS:
        raise KeyError(solver)
    if steps not in [int(count) for count in spec["step_counts"]]:
        raise ValueError(f"steps {steps} are not in the spec ladder")
    flow_names = tuple(str(name) for name in spec["flows"])
    held: dict[str, Any] = {}
    dtype_name: dict[str, str] = {}

    def build() -> Any:
        prepared = prepare_model(model, steps, spec)
        held["prepared"] = prepared
        return runner_for(prepared, solver)

    def call(runner: Any) -> Any:
        stash = {} if "name" not in dtype_name else None
        total = reduce_runner_outputs(runner, held["prepared"].parameters, flow_names, stash)
        if stash is not None:
            dtype_name["name"] = _dtype_name(stash["outputs"])
            for flow_name in flow_names:
                flow_dtype = _dtype_name(stash["flows"][flow_name])
                if flow_dtype != dtype_name["name"]:
                    dtype_name["name"] = f"outputs={dtype_name['name']}, {flow_name}={flow_dtype}"
        return total

    runner, timed = time_runner(build, call, n_warm=N_WARM)
    warm = timed.warm_s
    return _record(
        model=model,
        solver=solver,
        steps=steps,
        compartments=len(runner.model.compartments),
        build_s=timed.build_s,
        compile_s=timed.compile_s,
        warm_median_s=float(statistics.median(warm)),
        warm_s=list(warm),
        jax_version=jax.__version__,
        summerepi2_version=importlib.metadata.version("summerepi2"),
        dtype=dtype_name.get("name"),
        machine=machine,
        date=date,
        status="ok",
        reason=None,
        peak_rss_bytes=_peak_rss_bytes(),
    )


def run_matrix(path: Path) -> list[dict[str, Any]]:
    """Run every cell, writing ``path`` after each one. Returns the full list."""
    spec = load_spec()
    steps_list = [int(count) for count in spec["step_counts"]]
    machine = machine_description()
    date = _today()
    done = _load_ok(path)
    records: list[dict[str, Any]] = []
    stop_after: str | None = None
    stop_reason = ""

    print(
        f"matrix models={len(model_order())} solvers={len(SOLVERS)} "
        f"steps={steps_list} warm={N_WARM} timeout_s={cell_timeout_s()}",
        flush=True,
    )
    print(f"machine {machine}", flush=True)
    print(f"date {date}", flush=True)

    for model in model_order():
        for solver in SOLVERS:
            for steps in steps_list:
                key = (model, solver, steps)
                if stop_after is not None and model != stop_after:
                    records.append(_not_run(model, solver, steps, machine, date, stop_reason))
                    _write(path, records)
                    continue
                if key in done:
                    print(f"skip {model} {solver} {steps} already ok", flush=True)
                    record = done[key]
                else:
                    print(f"start {model} {solver} {steps}", flush=True)
                    record = _run_subprocess(model, solver, steps, machine, date)
                    print(
                        f"done {model} {solver} {steps} status={record['status']} "
                        f"build={record['build_s']} compile={record['compile_s']} "
                        f"warm_median={record['warm_median_s']} reason={record['reason']}",
                        flush=True,
                    )
                records.append(record)
                _write(path, records)
                if (model, solver, steps) == ("sir", "euler", 200):
                    try:
                        _require_spot_check(record)
                    except SystemExit as exc:
                        record["status"] = "failed"
                        record["reason"] = str(exc)
                        _write(path, records)
                        raise
                if record["status"] != "ok" and stop_after is None:
                    stop_after = model
                    stop_reason = f"{model} {solver} {steps}: {record['reason']}"
                    print(f"checkpoint: finish {model}, then stop. {stop_reason}", flush=True)
    _require_complete(records, steps_list)
    return records


def main(argv: list[str] | None = None) -> None:
    """Parent records the matrix. ``--cell`` times one cell and prints JSON."""
    parser = argparse.ArgumentParser(description="Record the summer2 timing matrix.")
    parser.add_argument("--cell", nargs=3, metavar=("MODEL", "SOLVER", "STEPS"))
    parser.add_argument("--machine", default=os.environ.get("SUMMER2_RECORD_MACHINE"))
    parser.add_argument("--date", default=os.environ.get("SUMMER2_RECORD_DATE"))
    args = parser.parse_args(argv)
    if args.cell is not None:
        model, solver, steps_text = args.cell
        machine = args.machine or machine_description()
        date = args.date or _today()
        try:
            record = execute_cell(model, solver, int(steps_text), machine, date)
        except Exception as exc:
            record = _record(
                model=model,
                solver=solver,
                steps=int(steps_text),
                compartments=None,
                build_s=None,
                compile_s=None,
                warm_median_s=None,
                warm_s=None,
                jax_version=jax.__version__,
                summerepi2_version=_package_version("summerepi2"),
                dtype=None,
                machine=machine,
                date=date,
                status="failed",
                reason=f"{type(exc).__name__}: {exc}",
                peak_rss_bytes=_peak_rss_bytes(),
            )
        _require_keys(record)
        print(json.dumps(record), flush=True)
        return
    records = run_matrix(RECORDED_PATH)
    ok = sum(1 for record in records if record["status"] == "ok")
    print(f"wrote {RECORDED_PATH} ok={ok} failed={len(records) - ok}", flush=True)


def _run_subprocess(model: str, solver: str, steps: int, machine: str, date: str) -> dict[str, Any]:
    env = os.environ.copy()
    env["SUMMER2_RECORD_MACHINE"] = machine
    env["SUMMER2_RECORD_DATE"] = date
    env["PYTHONUNBUFFERED"] = "1"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--cell",
        model,
        solver,
        str(steps),
        "--machine",
        machine,
        "--date",
        date,
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parent,
            env=env,
            capture_output=True,
            text=True,
            timeout=cell_timeout_s(),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return _failed_process(
            model,
            solver,
            steps,
            machine,
            date,
            f"timed out after {cell_timeout_s():.0f}s",
            exc.stderr,
        )
    parsed = _parse_child_stdout(completed.stdout)
    if parsed is not None and completed.returncode == 0:
        _require_keys(parsed)
        return parsed
    reason = _process_reason(completed.returncode)
    detail = (completed.stderr or completed.stdout or "").strip()
    if detail:
        reason = f"{reason}: {detail[-500:]}"
    if parsed is not None and parsed.get("status") == "failed":
        parsed["reason"] = parsed.get("reason") or reason
        _require_keys(parsed)
        return parsed
    return _failed_process(model, solver, steps, machine, date, reason, completed.stderr)


def _failed_process(
    model: str,
    solver: str,
    steps: int,
    machine: str,
    date: str,
    reason: str,
    stderr: str | None,
) -> dict[str, Any]:
    detail = (stderr or "").strip()
    if detail and detail not in reason:
        reason = f"{reason}: {detail[-500:]}"
    return _record(
        model=model,
        solver=solver,
        steps=steps,
        compartments=None,
        build_s=None,
        compile_s=None,
        warm_median_s=None,
        warm_s=None,
        jax_version=None,
        summerepi2_version=None,
        dtype=None,
        machine=machine,
        date=date,
        status="failed",
        reason=reason,
        peak_rss_bytes=None,
    )


def _not_run(
    model: str, solver: str, steps: int, machine: str, date: str, why: str
) -> dict[str, Any]:
    return _record(
        model=model,
        solver=solver,
        steps=steps,
        compartments=None,
        build_s=None,
        compile_s=None,
        warm_median_s=None,
        warm_s=None,
        jax_version=None,
        summerepi2_version=None,
        dtype=None,
        machine=machine,
        date=date,
        status="failed",
        reason=f"not run: stopped after {why}",
        peak_rss_bytes=None,
    )


def _record(**fields: Any) -> dict[str, Any]:
    record = {key: fields[key] for key in RECORD_KEYS}
    _require_keys(record)
    return record


def _require_keys(record: dict[str, Any]) -> None:
    missing = [key for key in RECORD_KEYS if key not in record]
    if missing:
        raise KeyError(f"record missing {missing}")


def _require_spot_check(record: dict[str, Any]) -> None:
    """``sir`` / euler / 200 must be a real warm call, not a second compile."""
    if record["status"] != "ok":
        raise SystemExit(f"spot check cell failed: {record['reason']}")
    if record["dtype"] != "float64":
        raise SystemExit(f"spot check dtype is {record['dtype']}, expected float64")
    warm = record["warm_median_s"]
    compile_s = record["compile_s"]
    if warm is None or compile_s is None or not warm < compile_s:
        raise SystemExit(
            f"spot check warm median {warm} is not smaller than compile {compile_s}; "
            "the timer is including compilation and the matrix must be rerun"
        )
    print(
        f"spot check sir euler 200 warm {warm} < compile {compile_s} dtype=float64",
        flush=True,
    )


def _require_complete(records: list[dict[str, Any]], steps_list: list[int]) -> None:
    expected = {
        (model, solver, steps)
        for model in model_order()
        for solver in SOLVERS
        for steps in steps_list
    }
    found = {(record["model"], record["solver"], record["steps"]) for record in records}
    if found != expected:
        missing = sorted(expected - found)
        raise SystemExit(f"matrix is missing cells: {missing}")
    for record in records:
        _require_keys(record)


def _load_ok(path: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    """Cells already recorded as ok. Failed cells are retried."""
    if not path.exists():
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, list):
        raise TypeError(f"{path} must be a JSON list")
    kept: dict[tuple[str, str, int], dict[str, Any]] = {}
    for record in loaded:
        if record.get("status") != "ok":
            continue
        _require_keys(record)
        kept[(record["model"], record["solver"], int(record["steps"]))] = record
    return kept


def _write(path: Path, records: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _parse_child_stdout(stdout: str) -> dict[str, Any] | None:
    for line in reversed(stdout.splitlines()):
        text = line.strip()
        if text.startswith("{") and text.endswith("}"):
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
    return None


def _process_reason(returncode: int) -> str:
    if returncode == 0:
        return "child exited 0 without a JSON record"
    if returncode < 0:
        number = -returncode
        try:
            name = signal.Signals(number).name
        except ValueError:
            name = str(number)
        return f"process killed by {name}"
    return f"process exited {returncode}"


def _dtype_name(value: Any) -> str:
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        return type(value).__name__
    name = getattr(dtype, "name", None)
    return str(name if name is not None else dtype)


def _peak_rss_bytes() -> int:
    """Peak resident set of this process. macOS reports bytes; Linux reports KiB."""
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return rss
    return rss * 1024


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _today() -> str:
    return datetime.date.today().isoformat()


def _sysctl(key: str) -> str | None:
    try:
        completed = subprocess.run(
            ["sysctl", "-n", key],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    text = completed.stdout.strip()
    return text or None


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:
        print(f"record failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
