from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TrialMetrics:
    subject: str
    ldlj: float
    sparc: float
    rmse: float


def _format_subject_label(subject: str) -> str:
    # Common in this repo: "subject_01" -> "S01"
    if subject.startswith("subject_"):
        suffix = subject.split("_", 1)[1]
        if suffix.isdigit():
            return f"S{int(suffix):02d}"
        return f"S{suffix}"
    if subject.lower().startswith("s") and len(subject) > 1:
        return subject
    return subject


def _as_float(x: Any) -> float:
    if x is None:
        return float("nan")
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _extract_trial_metrics_new_schema(
    trial: dict[str, Any],
    *,
    variant: str,
    ldlj_field: str,
    sparc_field: str,
    rmse_field: str,
) -> TrialMetrics:
    """
    Newer schema (e.g. quant_analysis_*.json):
      trial["metrics"][variant]["ldlj"][ldlj_field]
      trial["metrics"][variant]["sparc"][sparc_field]
      trial["metrics"][variant]["rmse"][rmse_field] (typically "overall")
      trial["trial_id"]["subject"]
    """
    subject = str(trial["trial_id"]["subject"])
    metrics = trial["metrics"][variant]

    ldlj = _as_float(metrics["ldlj"][ldlj_field])
    sparc = _as_float(metrics["sparc"][sparc_field])
    rmse = _as_float(metrics["rmse"][rmse_field])

    return TrialMetrics(subject=subject, ldlj=ldlj, sparc=sparc, rmse=rmse)


def _extract_trial_metrics_old_schema(
    trial: dict[str, Any],
    *,
    ldlj_source: str,
    sparc_source: str,
    rmse_field: str,
) -> TrialMetrics:
    """
    Older schema (e.g. quant_results_overall.json):
      trial["metrics"]["ldlj"][source]["ldlj"] where source in {"demo","dmp","delta"}
      trial["metrics"]["sparc"][source]["sparc"] where source in {"demo","dmp","delta"}
      trial["metrics"]["rmse"][rmse_field] where rmse_field typically "overall"
      trial["trial_id"]["subject"]
    """
    subject = str(trial["trial_id"]["subject"])
    metrics = trial["metrics"]

    ldlj = _as_float(metrics["ldlj"][ldlj_source]["ldlj"])
    sparc = _as_float(metrics["sparc"][sparc_source]["sparc"])
    rmse = _as_float(metrics["rmse"][rmse_field])

    return TrialMetrics(subject=subject, ldlj=ldlj, sparc=sparc, rmse=rmse)


def load_trial_metrics(
    json_path: Path,
    *,
    rmse_field: str,
    ldlj_source_old: str,
    sparc_source_old: str,
) -> list[TrialMetrics]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    trials: Iterable[dict[str, Any]] = data.get("trials", [])

    out: list[TrialMetrics] = []
    for t in trials:
        metrics = t.get("metrics", {})
        if not (isinstance(metrics, dict) and ("base" in metrics or "personalized" in metrics)):
            out.append(
                _extract_trial_metrics_old_schema(
                    t,
                    ldlj_source=ldlj_source_old,
                    sparc_source=sparc_source_old,
                    rmse_field=rmse_field,
                )
            )
            continue

        # New schema exists, but this function loads one series only:
        # keep legacy behavior by using "personalized" when present else "base",
        # and plotting "gen" fields.
        chosen_variant = "personalized" if "personalized" in metrics else "base"
        out.append(
            _extract_trial_metrics_new_schema(
                t,
                variant=chosen_variant,
                ldlj_field="gen",
                sparc_field="gen",
                rmse_field=rmse_field,
            )
        )

    return out


def load_trials_three_series(
    json_path: Path,
    *,
    rmse_field: str,
    ldlj_source_old: str,
    sparc_source_old: str,
) -> pd.DataFrame:
    """
    Returns a long-form dataframe with columns:
      subject, series in {"demo","base","personalized"}, ldlj, sparc, rmse

    - New schema: "demo" comes from metrics["base"] (or "personalized" if base missing)
                 "base" and "personalized" come from their respective "gen" fields.
    - Old schema: "demo" from demo, "base" from dmp (configurable), "personalized" = NaN.
    """
    data = json.loads(json_path.read_text(encoding="utf-8"))
    trials: Iterable[dict[str, Any]] = data.get("trials", [])

    rows: list[dict[str, Any]] = []
    for t in trials:
        subject = str(t["trial_id"]["subject"])
        metrics = t.get("metrics", {})

        if isinstance(metrics, dict) and ("base" in metrics or "personalized" in metrics):
            demo_from = "base" if "base" in metrics else "personalized"

            demo = _extract_trial_metrics_new_schema(
                t,
                variant=demo_from,
                ldlj_field="demo",
                sparc_field="demo",
                rmse_field=rmse_field,
            )
            base = (
                _extract_trial_metrics_new_schema(
                    t,
                    variant="base",
                    ldlj_field="gen",
                    sparc_field="gen",
                    rmse_field=rmse_field,
                )
                if "base" in metrics
                else TrialMetrics(subject=subject, ldlj=float("nan"), sparc=float("nan"), rmse=float("nan"))
            )
            personalized = (
                _extract_trial_metrics_new_schema(
                    t,
                    variant="personalized",
                    ldlj_field="gen",
                    sparc_field="gen",
                    rmse_field=rmse_field,
                )
                if "personalized" in metrics
                else TrialMetrics(subject=subject, ldlj=float("nan"), sparc=float("nan"), rmse=float("nan"))
            )

            rows.extend(
                [
                    {"subject": demo.subject, "series": "demo", "ldlj": demo.ldlj, "sparc": demo.sparc, "rmse": demo.rmse},
                    {"subject": base.subject, "series": "base", "ldlj": base.ldlj, "sparc": base.sparc, "rmse": base.rmse},
                    {
                        "subject": personalized.subject,
                        "series": "personalized",
                        "ldlj": personalized.ldlj,
                        "sparc": personalized.sparc,
                        "rmse": personalized.rmse,
                    },
                ]
            )
        else:
            demo = _extract_trial_metrics_old_schema(
                t, ldlj_source="demo", sparc_source="demo", rmse_field=rmse_field
            )
            base = _extract_trial_metrics_old_schema(
                t, ldlj_source=ldlj_source_old, sparc_source=sparc_source_old, rmse_field=rmse_field
            )
            rows.extend(
                [
                    {"subject": demo.subject, "series": "demo", "ldlj": demo.ldlj, "sparc": demo.sparc, "rmse": demo.rmse},
                    {"subject": base.subject, "series": "base", "ldlj": base.ldlj, "sparc": base.sparc, "rmse": base.rmse},
                    {"subject": base.subject, "series": "personalized", "ldlj": float("nan"), "sparc": float("nan"), "rmse": float("nan")},
                ]
            )

    return pd.DataFrame(rows)


def summarize_per_subject_three_series(df_long: pd.DataFrame) -> pd.DataFrame:
    if df_long.empty:
        return pd.DataFrame(columns=["subject", "series", "ldlj_mean", "sparc_mean", "rmse_mean", "n_trials"])

    summary = (
        df_long.groupby(["subject", "series"], as_index=False)
        .agg(
            ldlj_mean=("ldlj", "mean"),
            sparc_mean=("sparc", "mean"),
            rmse_mean=("rmse", "mean"),
            n_trials=("rmse", "size"),
        )
        .sort_values(["subject", "series"])
        .reset_index(drop=True)
    )
    return summary


def summarize_per_subject(rows: list[TrialMetrics]) -> pd.DataFrame:
    df = pd.DataFrame([r.__dict__ for r in rows])
    if df.empty:
        return pd.DataFrame(columns=["subject", "ldlj_mean", "sparc_mean", "rmse_mean", "n_trials"])

    summary = (
        df.groupby("subject", as_index=False)
        .agg(
            ldlj_mean=("ldlj", "mean"),
            sparc_mean=("sparc", "mean"),
            rmse_mean=("rmse", "mean"),
            n_trials=("rmse", "size"),
        )
        .sort_values("subject")
        .reset_index(drop=True)
    )

    return summary


def plot_per_subject_means_three_series(
    summary_long: pd.DataFrame,
    *,
    title: str | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    # Pivot to wide for plotting.
    subjects = sorted(summary_long["subject"].unique().tolist())
    subject_labels = [_format_subject_label(s) for s in subjects]
    x = np.arange(len(subjects))

    def wide(metric_col: str) -> pd.DataFrame:
        w = (
            summary_long.pivot(index="subject", columns="series", values=metric_col)
            .reindex(subjects)
            .reset_index(drop=True)
        )
        return w

    ldlj_w = wide("ldlj_mean")
    sparc_w = wide("sparc_mean")
    rmse_w = wide("rmse_mean")

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(max(10, len(subjects) * 0.85), 9), sharex=True)

    style = {
        "demo": {"label": "Demonstration", "color": "#4C78A8"},
        "base": {"label": "Base", "color": "#F58518"},
        "personalized": {"label": "Personalized", "color": "#54A24B"},
    }

    def plot_metric(ax: plt.Axes, w: pd.DataFrame, ylabel: str, subplot_title: str) -> None:
        for key in ["demo", "base", "personalized"]:
            if key not in w.columns:
                continue
            y = w[key].to_numpy(dtype=float)
            ax.plot(x, y, marker="o", linewidth=1.5, markersize=4, **style[key])
        ax.set_ylabel(ylabel)
        ax.set_title(subplot_title)
        ax.grid(True, alpha=0.35)
        ax.legend(loc="best", frameon=True)

    plot_metric(axes[0], ldlj_w, "LDLJ", "Mean LDLJ per Subject")
    plot_metric(axes[1], sparc_w, "SPARC", "Mean SPARC per Subject")
    plot_metric(axes[2], rmse_w, "RMSE", "Mean RMSE per Subject")
    axes[2].set_xlabel("Subject")

    axes[2].set_xticks(x)
    axes[2].set_xticklabels(subject_labels)

    if title:
        fig.suptitle(title)

    fig.tight_layout()
    return fig, axes


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Plot per-subject averages for LDLJ, SPARC, and RMSE from a results JSON."
    )
    p.add_argument(
        "--json",
        dest="json_path",
        type=Path,
        default=Path("results/quant_analysis_20260506_092902.json"),
        help="Path to the results JSON file.",
    )
    # Note: We always plot demo + base + personalized. These options only affect old-schema mapping.
    p.add_argument(
        "--rmse-field",
        choices=["overall"],
        default="overall",
        help="Which RMSE field to average.",
    )
    p.add_argument(
        "--ldlj-source-old",
        choices=["dmp", "demo", "delta"],
        default="dmp",
        help="(Old schema) Which LDLJ source to average.",
    )
    p.add_argument(
        "--sparc-source-old",
        choices=["dmp", "demo", "delta"],
        default="dmp",
        help="(Old schema) Which SPARC source to average.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="If set, save figure to this path instead of showing.",
    )
    p.add_argument("--dpi", type=int, default=200, help="DPI used when saving with --out.")
    p.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional plot title. Defaults to the JSON filename.",
    )
    return p


def main() -> None:
    args = build_argparser().parse_args()

    json_path: Path = args.json_path
    if not json_path.exists():
        raise FileNotFoundError(f"Results JSON not found: {json_path}")

    df_long = load_trials_three_series(
        json_path,
        rmse_field=args.rmse_field,
        ldlj_source_old=args.ldlj_source_old,
        sparc_source_old=args.sparc_source_old,
    )
    summary_long = summarize_per_subject_three_series(df_long)

    title = args.title if args.title is not None else "Mean LDJL, SPARC and RMSE per subject"
    fig, _axes = plot_per_subject_means_three_series(summary_long, title=title)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    else:
        plt.show()


if __name__ == "__main__":
    main()
