#!/usr/bin/env python3

import argparse
import csv
import re
import shlex
import statistics
import subprocess
import sys
from pathlib import Path


def run_command(command):
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )
    return result


def get_benchmark_help():
    result = run_command(
        [sys.executable, "-m", "mlx_lm.benchmark", "--help"]
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Unable to retrieve benchmark help:\n{result.stderr}"
        )

    return result.stdout + result.stderr


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
        ("KV", "kv_bits"),
        ("Prefill", "prefill_step"),
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


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark MLX-LM models and compare configurations."
    )

    parser.add_argument(
        "--model",
        default="mlx-community/Qwen3.5-9B-MLX-4bit",
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

    help_text = get_benchmark_help()

    # Discover supported CLI arguments.
    prompt_flag = find_flag(
        help_text,
        ["-p", "--prompt-tokens", "--prompt-length"],
    )

    generation_flag = find_flag(
        help_text,
        ["-g", "--generation-tokens", "--generation-length"],
    )

    kv_flag = find_flag(
        help_text,
        ["--kv-bits"],
    )

    prefill_flag = find_flag(
        help_text,
        ["--prefill-step-size"],
    )

    if not prompt_flag or not generation_flag:
        raise RuntimeError(
            "Could not identify prompt/generation flags.\n"
            "Inspect `mlx_lm.benchmark --help` and adapt the script."
        )

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

    for prompt_size in args.prompts:
        for kv_bits in args.kv_bits:
            for prefill_step in args.prefill:

                measurements = {
                    "prompt_tps": [],
                    "generation_tps": [],
                    "peak_memory_gb": [],
                }

                for repeat in range(args.repeats):
                    current += 1

                    command = [
                        sys.executable,
                        "-m",
                        "mlx_lm.benchmark",
                        "--model",
                        args.model,
                        prompt_flag,
                        str(prompt_size),
                        generation_flag,
                        str(args.generation),
                    ]

                    if kv_bits.lower() != "none":
                        if not kv_flag:
                            print(
                                "Skipping KV configuration: "
                                "installed benchmark has no --kv-bits"
                            )
                            break

                        command.extend([
                            kv_flag,
                            kv_bits,
                        ])

                    if prefill_flag:
                        command.extend([
                            prefill_flag,
                            str(prefill_step),
                        ])

                    print(
                        f"[{current}/{total}] "
                        f"Prompt={prompt_size}, "
                        f"KV={kv_bits}, "
                        f"Prefill={prefill_step}, "
                        f"Run={repeat + 1}"
                    )

                    result = run_command(command)

                    log_name = (
                        f"p{prompt_size}_kv{kv_bits}_"
                        f"prefill{prefill_step}_run{repeat + 1}.txt"
                    )

                    log_path = log_dir / log_name
                    log_path.write_text(
                        "COMMAND:\n"
                        + shlex.join(command)
                        + "\n\nSTDOUT:\n"
                        + result.stdout
                        + "\n\nSTDERR:\n"
                        + result.stderr
                    )

                    if result.returncode != 0:
                        print(f"  FAILED. See {log_path}")
                        continue

                    metrics = parse_metrics(
                        result.stdout + "\n" + result.stderr
                    )

                    for key, value in metrics.items():
                        if value is not None:
                            measurements[key].append(value)

                def average(key):
                    values = measurements[key]
                    return (
                        statistics.mean(values)
                        if values else None
                    )

                def deviation(key):
                    values = measurements[key]
                    return (
                        statistics.stdev(values)
                        if len(values) > 1 else None
                    )

                rows.append({
                    "model": args.model.split("/")[-1],
                    "prompt_tokens": prompt_size,
                    "generation_tokens": args.generation,
                    "kv_bits": kv_bits,
                    "prefill_step": prefill_step,
                    "prompt_tps": average("prompt_tps"),
                    "prompt_tps_std": deviation("prompt_tps"),
                    "generation_tps": average("generation_tps"),
                    "generation_tps_std": deviation("generation_tps"),
                    "peak_memory_gb": average("peak_memory_gb"),
                })

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
