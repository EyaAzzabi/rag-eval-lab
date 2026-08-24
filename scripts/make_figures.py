"""Render the figures used in the README from results/results.json."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "results.json"
FIGURES = ROOT / "figures"

# Colour-blind-safe, and distinguishable in greyscale when printed.
COLOURS = {"bm25": "#0072B2", "dense": "#E69F00", "hybrid": "#009E73"}
RETRIEVERS = ["bm25", "dense", "hybrid"]


def load() -> list[dict]:
    if not RESULTS.exists():
        raise SystemExit(f"{RESULTS} not found. Run: python -m rageval.evaluate")
    return json.loads(RESULTS.read_text(encoding="utf-8"))["runs"]


def figure_quality_by_strategy(runs: list[dict]) -> None:
    chunkings = sorted({r["chunking"] for r in runs})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    for ax, metric in zip(axes, ["ndcg@10", "recall@10"], strict=True):
        width = 0.26
        for i, retriever in enumerate(RETRIEVERS):
            values = [
                next(
                    (
                        r["metrics"][metric]
                        for r in runs
                        if r["retriever"] == retriever and r["chunking"] == c
                    ),
                    0.0,
                )
                for c in chunkings
            ]
            positions = [x + (i - 1) * width for x in range(len(chunkings))]
            bars = ax.bar(
                positions, values, width, label=retriever, color=COLOURS[retriever]
            )
            ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)

        ax.set_xticks(range(len(chunkings)))
        ax.set_xticklabels(chunkings)
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} by retriever and chunking")
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)

    axes[0].legend(title="retriever", frameon=False)
    fig.suptitle("SciFact, 300 judged queries", y=1.02, fontsize=10, color="#555")
    fig.tight_layout()
    fig.savefig(FIGURES / "quality_by_strategy.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def figure_quality_vs_latency(runs: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    markers = {"whole": "o", "sentence3": "s", "window120": "^"}

    for run in runs:
        ax.scatter(
            run["latency_ms_p50"],
            run["metrics"]["ndcg@10"],
            s=110,
            color=COLOURS[run["retriever"]],
            marker=markers.get(run["chunking"], "o"),
            edgecolor="white",
            linewidth=1.2,
            zorder=3,
        )
        ax.annotate(
            f"{run['retriever']}/{run['chunking']}",
            (run["latency_ms_p50"], run["metrics"]["ndcg@10"]),
            textcoords="offset points",
            xytext=(7, 4),
            fontsize=7.5,
            color="#444",
        )

    ax.set_xscale("log")
    ax.set_xlabel("median query latency (ms, log scale)")
    ax.set_ylabel("nDCG@10")
    ax.set_title("Retrieval quality against what it costs to serve")
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIGURES / "quality_vs_latency.png", dpi=160)
    plt.close(fig)


def figure_recall_by_k(runs: list[dict]) -> None:
    """How deep you must retrieve before the answer is present at all."""
    ks = [1, 3, 5, 10]
    best_chunking = "whole"
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for retriever in RETRIEVERS:
        run = next(
            (r for r in runs if r["retriever"] == retriever and r["chunking"] == best_chunking),
            None,
        )
        if not run:
            continue
        ax.plot(
            ks,
            [run["metrics"][f"recall@{k}"] for k in ks],
            marker="o",
            color=COLOURS[retriever],
            label=retriever,
            linewidth=2,
        )
    ax.set_xticks(ks)
    ax.set_xlabel("k (documents retrieved)")
    ax.set_ylabel("recall@k")
    ax.set_title(f"Recall against retrieval depth ({best_chunking} chunking)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIGURES / "recall_by_k.png", dpi=160)
    plt.close(fig)


def figure_chunking_divergence(runs: list[dict]) -> None:
    """The headline finding: finer chunks help dense retrieval and hurt BM25.

    Strategies are ordered by chunk count (coarse to fine) rather than by name.
    That ordering is what makes the effect legible -- both curves become
    monotonic, and they cross.
    """
    order = [c for _, c in sorted({(r["n_chunks"], r["chunking"]) for r in runs})]
    xs = range(len(order))

    fig, ax = plt.subplots(figsize=(8, 5))
    for retriever in RETRIEVERS:
        ys = [
            next(
                r["metrics"]["ndcg@10"]
                for r in runs
                if r["retriever"] == retriever and r["chunking"] == c
            )
            for c in order
        ]
        style = "--" if retriever == "hybrid" else "-"
        ax.plot(
            xs, ys, marker="o", markersize=8, linewidth=2.5, linestyle=style,
            color=COLOURS[retriever], label=retriever, zorder=3,
        )
        for x, y in zip(xs, ys, strict=True):
            ax.annotate(
                f"{y:.3f}", (x, y), textcoords="offset points",
                xytext=(0, 11), ha="center", fontsize=8, color=COLOURS[retriever],
            )

    # Headroom so the value labels above each marker are not clipped.
    ax.margins(y=0.16)

    # Mark where lexical and dense retrieval swap places.
    ax.axvspan(0, 1, color="#999", alpha=0.08, zorder=0)
    ax.text(
        0.5, ax.get_ylim()[0] + 0.002, "they cross here",
        ha="center", fontsize=8.5, color="#555", style="italic",
    )

    counts = {c: next(r["n_chunks"] for r in runs if r["chunking"] == c) for c in order}
    ax.set_xticks(list(xs))
    nl = chr(10)
    ax.set_xticklabels([f"{c}{nl}{counts[c]:,} chunks" for c in order])
    ax.set_xlabel("chunking strategy, coarse to fine")
    ax.set_ylabel("nDCG@10")
    ax.set_title("Finer chunks help dense retrieval and hurt BM25")
    ax.legend(frameon=False, loc="center right")
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(FIGURES / "chunking_divergence.png", dpi=160)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    runs = load()
    figure_chunking_divergence(runs)
    figure_quality_by_strategy(runs)
    figure_quality_vs_latency(runs)
    figure_recall_by_k(runs)
    print(f"Wrote 4 figures to {FIGURES}")


if __name__ == "__main__":
    main()
