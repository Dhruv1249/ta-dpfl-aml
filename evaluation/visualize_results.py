from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def generate_result_plots(output_dir: str | Path = "outputs"):
    output_dir = Path(output_dir)
    overall_path = output_dir / "overall_results.csv"
    typology_path = output_dir / "typology_results.csv"
    if not overall_path.exists():
        return []
    overall = pd.read_csv(overall_path)
    created = []

    if not overall.empty and {"method", "AP"}.issubset(overall.columns):
        summary = overall.groupby("method", as_index=False)["AP"].mean()
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(summary["method"], summary["AP"])
        ax.set_ylabel("Average Precision")
        ax.set_title("Overall AML detection AP by method")
        ax.tick_params(axis="x", labelrotation=25)
        fig.tight_layout()
        path = output_dir / "overall_ap_by_method.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        created.append(str(path))

    if not overall.empty and {"global_epsilon", "AP", "method"}.issubset(overall.columns):
        private = overall.dropna(subset=["global_epsilon", "AP"])
        fig, ax = plt.subplots(figsize=(7, 5))
        if len(private):
            ax.scatter(private["global_epsilon"], private["AP"])
            for _, row in private.iterrows():
                ax.annotate(str(row["method"]), (row["global_epsilon"], row["AP"]), fontsize=8)
        else:
            ax.text(0.5, 0.5, "No private method results", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("Worst-case record-level epsilon")
        ax.set_ylabel("Average Precision")
        ax.set_title("Privacy–utility comparison (interpret with cap-matching report)")
        fig.tight_layout()
        path = output_dir / "privacy_utility_comparison.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        created.append(str(path))

    if typology_path.exists():
        per_typology = pd.read_csv(typology_path)
        if not per_typology.empty and {"typology", "method", "AP"}.issubset(per_typology.columns):
            pivot = per_typology.pivot_table(index="typology", columns="method", values="AP", aggfunc="mean")
            fig, ax = plt.subplots(figsize=(12, 6))
            pivot.plot(kind="bar", ax=ax)
            ax.set_ylabel("Average Precision (typology-positive vs BG)")
            ax.set_title("Typology-specific detection AP")
            ax.tick_params(axis="x", labelrotation=30)
            fig.tight_layout()
            path = output_dir / "typology_performance.png"
            fig.savefig(path, dpi=160)
            plt.close(fig)
            created.append(str(path))
    return created
