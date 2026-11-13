#!/usr/bin/env python3

import argparse
import csv
import re
import shlex
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


class BenchmarkError(RuntimeError):
    pass


class BenchmarkResult:
    def __init__(self, returncode, stdout, stderr, timed_out=False):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.timed_out = timed_out


def run_command(command, timeout=900.0):
    """Run a command and preserve partial output on timeout."""
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        return BenchmarkResult(process.returncode, stdout, stderr)
    except subprocess.TimeoutExpired:
        timed_out = True
        process.kill()
        stdout, stderr = process.communicate()
        return BenchmarkResult(
            process.returncode,
            stdout or "",
            stderr or "",
            timed_out,
        )


class Backend:
    name = "base"

    def command(self, settings):
        raise NotImplementedError


class MlxLmBackend(Backend):
    name = "mlx"
    module = "mlx_lm.benchmark"

    def discover_flags(self):
        result = run_command(
            [sys.executable, "-m", self.module, "--help"]
        )
        if result.returncode != 0:
            raise BenchmarkError(
                f"Unable to retrieve {self.module} help:\n{result.stderr}"
            )
        return result.stdout + "\n" + result.stderr

    def command(self, settings):
        flags = settings["flags"]
        command = [sys.executable, "-m", self.module, "--model", settings["model"]]
        if not flags["prompt"] or not flags["generation"]:
            raise BenchmarkError("Could not identify prompt/generation flags.")
        command.extend([flags["prompt"], str(settings["prompt_tokens"])])
        command.extend([flags["generation"], str(settings["generation_tokens"])])
        if settings["kv_bits"].lower() != "none":
            if not flags["kv_bits"]:
                raise BenchmarkError("Installed benchmark has no --kv-bits.")
            command.extend([flags["kv_bits"], settings["kv_bits"]])
        if settings["prefill_step"] is not None:
            if not flags["prefill_step"]:
                raise BenchmarkError("Installed benchmark has no prefill flag.")
            command.extend([flags["prefill_step"], str(settings["prefill_step"])])
        if settings["mtp"] is not None:
            if not flags["mtp"]:
                raise BenchmarkError("Installed benchmark has no MTP flag.")
            command.extend([flags["mtp"], str(settings["mtp"])])
        return command


BACKENDS = {"mlx": MlxLmBackend}


def find_flag(help_text, candidates):
    """Find the first supported CLI flag."""
    for flag in candidates:
        if re.search(
            rf"(?<![\w-]){re.escape(flag)}(?:[=,\s]|$)",
            help_text,
        ):
            return flag
    return None


def parse_metrics(output):
    """Extract metrics from MLX-LM benchmark output."""

    patterns = {
        "prompt_tps": [
            r"prompt_tps\s*[:=]\s*([\d.]+)",
            r"prompt.*?([\d.]+)\s*tokens?/s",
            r"prefill.*?([\d.]+)\s*tokens?/s",
        ],
        "generation_tps": [
            r"generation_tps\s*[:=]\s*([\d.]+)",
            r"generation.*?([\d.]+)\s*tokens?/s",
            r"decode.*?([\d.]+)\s*tokens?/s",
        ],
        "peak_memory_gb": [
            r"peak_memory(?:_gb)?\s*[:=]\s*([\d.]+)",
            r"peak memory\s*[:=]?\s*([\d.]+)\s*GB",
            r"peak memory\s*[:=]?\s*([\d.]+)\s*GiB",
        ],
    }

    metrics = {}

    for metric, expressions in patterns.items():
        value = None

        for expression in expressions:
            match = re.search(
                expression,
                output,
                re.IGNORECASE,
            )

            if match:
                value = float(match.group(1))
                break

        metrics[metric] = value

    return metrics


def format_value(value, suffix=""):
    if value is None:
        return "N/A"
    return f"{value:.2f}{suffix}"


def print_table(rows):
    columns = [
        ("Model", "model"),
        ("Status", "status"),
        ("KV", "kv_bits"),
        ("Prefill", "prefill_step"),
        ("MTP", "mtp"),
        ("Prompt TPS", "prompt_tps"),
        ("Generation TPS", "generation_tps"),
        ("Peak GB", "peak_memory_gb"),
    ]

    widths = {
        name: max(
            len(name),
            max(
                (len(str(row.get(key, ""))) for row in rows),
                default=0,
            ),
        )
        for name, key in columns
    }

    header = " | ".join(
        name.ljust(widths[name])
        for name, _ in columns
    )

    print("\n" + header)
    print("-" * len(header))

    for row in rows:
        values = []

        for name, key in columns:
            value = row.get(key)

            if key in ("prompt_tps", "generation_tps"):
                value = format_value(value)

            elif key == "peak_memory_gb":
                value = format_value(value, " GB")

            elif value is None:
                value = "None"

            values.append(str(value).ljust(widths[name]))

        print(" | ".join(values))


def save_results(rows, csv_path):
    """Write completed and failed benchmark rows to CSV."""
    csv_path = Path(csv_path)
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    else:
        csv_path.write_text("", encoding="utf-8")
    return csv_path


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark MLX-LM models and compare configurations."
    )

    parser.add_argument(
        "--backend",
        choices=sorted(BACKENDS),
        default="mlx",
    )

    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Model identifier; repeat to compare models.",
    )

    parser.add_argument(
        "--prompts",
        nargs="+",
        type=int,
        default=[512, 2048, 8192],
    )

    parser.add_argument(
        "--generation",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--kv-bits",
        nargs="+",
        default=["4", "none"],
        help="KV cache configurations, e.g. 4 8 none",
    )

    parser.add_argument(
        "--prefill",
        nargs="+",
        type=int,
        default=[512, 1024, 2048],
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--mtp",
        action="append",
        default=[],
        help="MTP values to test; use none to disable.",
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=900.0,
        help="Per-benchmark-process timeout in seconds (default: 900).",
    )

    parser.add_argument(
        "--csv",
        default="mlx_benchmark_results.csv",
    )

    parser.add_argument(
        "--logs",
        default="mlx_benchmark_logs",
    )

    args = parser.parse_args()

    if args.repeats < 1:
        parser.error("--repeats must be at least 1")
    if not args.model:
        parser.error("--model is required (repeat it to compare models)")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if any(size <= 0 for size in args.prompts):
        parser.error("--prompts values must be positive")
    if args.generation <= 0:
        parser.error("--generation must be positive")
    if args.mtp:
        for value in args.mtp:
            if value.lower() != "none" and not value.isdigit():
                parser.error("--mtp accepts integers or none")
    else:
        args.mtp = ["none"]
    if not args.prefill:
        parser.error("--prefill must contain at least one value")

    backend = BACKENDS[args.backend]()
    flags = {
        "prompt": find_flag(
            backend.discover_flags(),
            ["-p", "--prompt-tokens", "--prompt-length"],
        ),
        "generation": find_flag(
            backend.discover_flags(),
            ["-g", "--generation-tokens", "--generation-length"],
        ),
        "kv_bits": find_flag(
            backend.discover_flags(),
            ["--kv-bits"],
        ),
        "prefill_step": find_flag(
            backend.discover_flags(),
            ["--prefill-step-size"],
        ),
        "mtp": find_flag(
            backend.discover_flags(),
            ["--mtp", "--num-mtp", "--mtp-tokens"],
        ),
    }

    log_dir = Path(args.logs)
    log_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    total = (
        len(args.prompts)
        * len(args.kv_bits)
        * len(args.prefill)
        * args.repeats
    )

    current = 0

    configs = [
        (model, prompt_size, kv_bits, prefill_step, mtp_value)
        for model in args.model
        for prompt_size in args.prompts
        for kv_bits in args.kv_bits
        for prefill_step in args.prefill
        for mtp_value in args.mtp
    ]
    total = len(configs) * args.repeats

    try:
        for index, (model, prompt_size, kv_bits, prefill_step, mtp_value) in enumerate(
            configs, start=1
        ):
            measurements = {
                "prompt_tps": [],
                "generation_tps": [],
                "peak_memory_gb": [],
            }
            settings = {
                "model": model,
                "prompt_tokens": prompt_size,
                "generation_tokens": args.generation,
                "kv_bits": kv_bits,
                "prefill_step": prefill_step,
                "mtp": None if mtp_value.lower() == "none" else int(mtp_value),
                "flags": flags,
            }
            base_row = {
                "model": model,
                "backend": args.backend,
                "prompt_tokens": prompt_size,
                "generation_tokens": args.generation,
                "kv_bits": kv_bits,
                "prefill_step": prefill_step,
                "mtp": mtp_value,
                "status": "FAILED",
                "error": "",
                "prompt_tps": None,
                "prompt_tps_std": None,
                "generation_tps": None,
                "generation_tps_std": None,
                "peak_memory_gb": None,
                "timestamp_utc": "",
            }

            try:
                command = backend.command(settings)
            except BenchmarkError as error:
                base_row["error"] = str(error)
                base_row["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
                rows.append(base_row)
                continue

            for repeat in range(args.repeats):
                current += 1
                progress = (
                    f"[{current}/{total}] Prompt={prompt_size}, "
                    f"KV={kv_bits}, Prefill={prefill_step}, "
                    f"MTP={mtp_value}, Run={repeat + 1}"
                )
                print(progress)
                result = run_command(command, args.timeout)
                log_path = log_dir / (
                    f"run_{index:03d}_{repeat + 1:02d}_p{prompt_size}_"
                    f"kv{kv_bits}_prefill{prefill_step}_mtp{mtp_value}.txt"
                )
                log_path.write_text(
                    "COMMAND:\n"
                    + shlex.join(command)
                    + "\n\nSTDOUT:\n"
                    + result.stdout
                    + "\n\nSTDERR:\n"
                    + result.stderr
                    + ("\n\nTIMEOUT" if result.timed_out else ""),
                    encoding="utf-8",
                )

                if result.timed_out:
                    base_row["status"] = "TIMEOUT"
                    base_row["error"] = f"Exceeded {args.timeout:g}s timeout."
                    print(f"  TIMEOUT after {args.timeout:g}s. See {log_path}")
                    break
                if result.returncode != 0:
                    print(f"  FAILED. See {log_path}")
                    continue

                metrics = parse_metrics(result.stdout + "\n" + result.stderr)
                for key, value in metrics.items():
                    if value is not None:
                        measurements[key].append(value)

            def average(key):
                values = measurements[key]
                return statistics.mean(values) if values else None

            def deviation(key):
                values = measurements[key]
                return statistics.stdev(values) if len(values) > 1 else None

            if measurements["prompt_tps"] or measurements["generation_tps"]:
                base_row["status"] = "OK"
                base_row.update({
                    "prompt_tps": average("prompt_tps"),
                    "prompt_tps_std": deviation("prompt_tps"),
                    "generation_tps": average("generation_tps"),
                    "generation_tps_std": deviation("generation_tps"),
                    "peak_memory_gb": average("peak_memory_gb"),
                })
            if not base_row["error"]:
                base_row["error"] = "No benchmark metrics were parsed."
            base_row["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
            rows.append(base_row)
    except KeyboardInterrupt:
        print("\nInterrupted; preserving completed results.")

    if not rows:
        print("No test results were collected.")
        return 1

    print_table(rows)
    output_path = save_results(rows, args.csv)
    print(f"\nResults saved to: {output_path}")
    print(f"Raw logs saved to: {log_dir.resolve()}")
    return 0

    # Display results.
    print_table(rows)

    # Export CSV.
    if rows:
        with open(args.csv, "w", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=rows[0].keys(),
            )
            writer.writeheader()
            writer.writerows(rows)

        print(f"\nCSV saved to: {args.csv}")
        print(f"Raw logs saved to: {log_dir}/")


if __name__ == "__main__":
    main()
