"""Plot verified timings and mark failures separately from measured runtimes."""

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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metric", choices=("summaries", "full"), default="summaries")
    args = parser.parse_args()
    results = json.loads(args.results.read_text())
    sizes, limits = results["sizes"], results["limits_gib"]
    fig, axes = plt.subplots(
        2,
        len(limits),
        figsize=(max(7.8, 4.5 * len(limits)), 5.4),
        sharex="col",
        sharey="row",
        squeeze=False,
        gridspec_kw={"height_ratios": (4, 1)},
    )
    names = {
        "scipy": "NumPy / SciPy",
        "duckdb": "dbnumpy DuckDB",
        "datafusion": "dbnumpy DataFusion",
    }
    colors = {"scipy": "#343d46", "duckdb": "#197a72", "datafusion": "#bc5925"}
    for column, ceiling in enumerate(limits):
        ax, failures = axes[:, column]
        for engine_index, engine in enumerate(names):
            times = []
            failed_sizes = []
            for size in sizes:
                rows = [
                    r
                    for r in results["trials"]
                    if r["engine"] == engine
                    and r["cells"] == size
                    and r["limit_gib"] == ceiling
                ]
                if len(rows) != 1:
                    raise ValueError("This pilot figure expects one trial per point")
                row = rows[0]
                verification = row.get("verification")
                valid = isinstance(verification, dict) and verification.get(
                    "scope"
                ) in ("summaries_only", "full_output_and_summaries")
                if args.metric == "full":
                    valid = (
                        valid
                        and row["status"] == "passed"
                        and verification["scope"] == "full_output_and_summaries"
                    )
                times.append(
                    (
                        row["total_seconds"]
                        if args.metric == "full"
                        else row["input_open_seconds"] + row["workflow_seconds"]
                    )
                    if valid
                    else np.nan
                )
                if not valid:
                    failed_sizes.append(size)
            ax.plot(
                sizes,
                times,
                marker="o",
                markersize=4,
                linewidth=1.7,
                label=names[engine],
                color=colors[engine],
            )
            failures.scatter(
                failed_sizes,
                [engine_index] * len(failed_sizes),
                marker="x",
                s=32,
                linewidths=1.5,
                color=colors[engine],
            )
        full_ram = results.get("requested_limits") == ["max"]
        ax.set_title(
            f"{ceiling:.3g} GiB ceiling" + (" (full WSL RAM)" if full_ram else ""),
            fontsize=11,
        )
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 3)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        failures.set_xlabel("Input cells")
        failures.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:g}k"))
        failures.set_xlim(0, max(sizes) * 1.025)
        failures.set_ylim(2.6, -0.6)
        failures.set_yticks(range(3), ["SciPy", "DuckDB", "DataFusion"], fontsize=8)
        failures.tick_params(axis="y", length=0)
        failures.spines[["top", "right", "left"]].set_visible(False)
        failures.grid(axis="y", alpha=0.12)
        ax.grid(axis="y", which="major", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("Seconds (log scale)")
    axes[1, 0].set_ylabel("Failed (×)", fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.07),
        fontsize=9,
    )
    fig.suptitle(
        "Mouse brain preprocessing: "
        + (
            "full workflow and sparse export"
            if args.metric == "full"
            else "time to verified gene summaries"
        ),
        fontsize=12,
        y=0.99,
    )
    fig.text(
        0.5,
        0.012,
        "Includes input opening. "
        + (
            "Full matrix export included."
            if args.metric == "full"
            else "Full matrix export excluded."
        )
        + "\n× marks a failed result, not a runtime. One trial per point.",
        ha="center",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.16, 1, 0.94))
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
