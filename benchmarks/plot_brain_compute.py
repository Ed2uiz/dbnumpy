"""Plot full workflow runtime, including native materialization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--datafusion-results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = json.loads(args.results.read_text())
    assert results["endpoint"] == "native"
    assert results["repeats"] == 1 and len(results["limits_gib"]) == 1
    updated = None
    if args.datafusion_results is not None:
        updated = json.loads(args.datafusion_results.read_text())
        assert updated["endpoint"] == "native" and updated["repeats"] == 1
        assert updated["workflow_sha256"] == results["workflow_sha256"]
        assert updated["dataset"] == results["dataset"]
        assert updated["versions"] == results["versions"]
        assert set(updated["sizes"]).issubset(results["sizes"])
        assert all(row["engine"] == "datafusion" for row in updated["trials"])
        for size in updated["sizes"]:
            assert (
                updated["reference_sha256"][str(size)]
                == results["reference_sha256"][str(size)]
            )
    names = {
        "scipy": "NumPy / SciPy",
        "duckdb": "dbnumpy DuckDB",
        "datafusion": "dbnumpy DataFusion",
    }
    colors = {"scipy": "#343d46", "duckdb": "#197a72", "datafusion": "#bc5925"}
    fig, (ax, failures) = plt.subplots(
        2,
        1,
        figsize=(7, 4.8),
        sharex=True,
        layout="constrained",
        gridspec_kw={"height_ratios": (10, 1)},
    )
    sizes = results["sizes"]
    failure_labels = []
    for engine in names:
        measurements = (
            updated if engine == "datafusion" and updated is not None else results
        )
        engine_sizes = measurements["sizes"]
        times, failed = [], []
        for size in engine_sizes:
            rows = [
                row
                for row in measurements["trials"]
                if row["engine"] == engine and row["cells"] == size
            ]
            assert len(rows) == 1
            row = rows[0]
            verification = row.get("verification")
            valid = (
                isinstance(verification, dict)
                and verification["scope"] == "native_summaries_count_and_sample"
                and row["status"] == "passed"
            )
            times.append(row["total_seconds"] if valid else np.nan)
            if not valid:
                failed.append(size)
        ax.plot(
            engine_sizes,
            times,
            color=colors[engine],
            label=names[engine],
            marker="o",
            markersize=4.5,
            linewidth=2,
        )
        if failed:
            failures.scatter(
                failed,
                [len(failure_labels)] * len(failed),
                marker="x",
                s=38,
                linewidths=1.7,
                color=colors[engine],
            )
            short_name = {
                "scipy": "SciPy",
                "duckdb": "DuckDB",
                "datafusion": "DataFusion",
            }
            failures_are_oom = all(
                row["status"] == "memory_limit"
                for row in measurements["trials"]
                if row["engine"] == engine and row["cells"] in failed
            )
            label = "OOM" if failures_are_oom else "failed"
            failure_labels.append(f"{short_name[engine]} {label}")
    ax.set_title(
        "Single-cell NumPy Transformations Across Backends",
        loc="center",
        fontsize=12,
        fontweight="bold",
        pad=14,
    )
    ax.set_ylabel("Total runtime (s, log scale)")
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 3)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(axis="y", which="major", alpha=0.16)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    failures.set_xlabel("Input cells")
    failures.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:g}k"))
    failures.set_xlim(0, max(sizes) * 1.025)
    failures.set_ylim(max(len(failure_labels) - 0.4, 0.6), -0.6)
    failures.set_yticks(range(len(failure_labels)), failure_labels, fontsize=8)
    failures.tick_params(axis="y", length=0, pad=8)
    failures.spines[["top", "right", "left"]].set_visible(False)
    failures.grid(axis="y", alpha=0.1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(
            args.output.with_suffix(f".{extension}"),
            dpi=180,
            bbox_inches="tight",
            facecolor="white",
        )


if __name__ == "__main__":
    main()
