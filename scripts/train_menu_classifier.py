from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import numpy as np

_scripts_dir = Path(__file__).resolve().parent
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))

# Project root = parent of scripts/ (datasets under data/ by default)
PROJECT_ROOT = _scripts_dir.parent

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

from category_standardization import standardize_category


def save_confusion_matrix_heatmap(
    cm: np.ndarray,
    labels: list[str],
    path: Path,
    *,
    dpi: int = 150,
) -> None:
    """Write a seaborn heatmap of the confusion matrix (counts)."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError as e:
        raise ImportError(
            "Confusion matrix heatmap requires matplotlib and seaborn. "
            "Install: pip install matplotlib seaborn"
        ) from e

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Readable figure for long class names
    n = len(labels)
    fig_w = max(10.0, 0.55 * n)
    fig_h = max(8.0, 0.5 * n)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=labels,
        yticklabels=labels,
        cmap="Blues",
        ax=ax,
        square=False,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def print_top_confusion_pairs(cm: np.ndarray, labels: list[str], k: int = 8) -> None:
    """Largest off-diagonal cells (true row -> pred col); useful for report captions."""
    pairs: list[tuple[int, str, str]] = []
    for i, ti in enumerate(labels):
        for j, pj in enumerate(labels):
            if i != j and cm[i, j] > 0:
                pairs.append((int(cm[i, j]), ti, pj))
    pairs.sort(reverse=True, key=lambda x: x[0])
    print(f"\nTop {k} misclassification counts (true row → predicted col):")
    for count, ti, pj in pairs[:k]:
        print(f"  {count:5d}  {ti}  →  {pj}")


def resolve_report_dir(report_dir_arg: Path | None) -> Path:
    """Default: project_root/results/. Override with --report-dir."""
    if report_dir_arg is not None:
        return Path(report_dir_arg).resolve()
    return (PROJECT_ROOT / "results").resolve()


def resolve_data_csv(path_str: str) -> Path:
    """Resolve training CSV: cwd first, then project root."""
    p = Path(path_str)
    if p.is_absolute():
        return p
    c1 = (Path.cwd() / p).resolve()
    if c1.exists():
        return c1
    c2 = (PROJECT_ROOT / p).resolve()
    return c2


def write_evaluation_csvs(
    report_dir: Path,
    base: str,
    run_id: int,
    summary_rows: list[dict[str, object]],
    per_class_parts: list[pd.DataFrame],
    dist: pd.Series,
) -> tuple[Path, Path, Path]:
    """Write summary, per-class, and class-distribution CSVs; return absolute paths."""
    report_dir.mkdir(parents=True, exist_ok=True)
    rid = int(run_id)
    sum_path = report_dir / f"{base}_summary_{rid}.csv"
    pc_path = report_dir / f"{base}_per_class_{rid}.csv"
    dist_path = report_dir / f"{base}_class_distribution_{rid}.csv"
    pd.DataFrame(summary_rows).to_csv(sum_path, index=False, encoding="utf-8-sig")
    pd.concat(per_class_parts, ignore_index=True).to_csv(
        pc_path, index=False, encoding="utf-8-sig"
    )
    dist.rename_axis("class").reset_index(name="count").to_csv(
        dist_path, index=False, encoding="utf-8-sig"
    )
    return sum_path.resolve(), pc_path.resolve(), dist_path.resolve()


def build_text_features(df: pd.DataFrame) -> pd.Series:
    """Combine key text fields (focused, low-noise features)."""
    return (
        df["item_name_original"].fillna("") + " " +
        df["item_name_english"].fillna("")
    ).str.strip()


def load_and_clean_data(csv_path: str) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Load CSV and filter rows for classification."""
    df = pd.read_csv(csv_path)

    # Keep rows with category + some text
    df = df.fillna("")
    df["text_features"] = build_text_features(df)

    df = df[
        (df["category_name_original"].str.strip() != "") &
        (df["text_features"].str.strip() != "")
    ].copy()

    # Map fine-grained labels → standardized buckets (see category_standardization.py)
    df["category_standardized"] = df["category_name_original"].apply(standardize_category)

    # Drop standardized classes with too few samples for stable CV (StratifiedKFold needs k <= n per class)
    category_counts = df["category_standardized"].value_counts()
    valid_categories = category_counts[category_counts >= 5].index
    df = df[df["category_standardized"].isin(valid_categories)].copy()

    X = df["text_features"]
    y = df["category_standardized"]

    return X, y, df


def print_other_original_counts(df: pd.DataFrame, head_n: int = 50) -> None:
    """Diagnostic: top raw category_name_original values that map to Other (post CV filter)."""
    sub = df[df["category_standardized"] == "Other"]["category_name_original"]
    print(f'\nTop {head_n} category_name_original for standardized "Other" (n={len(sub)}):')
    print(sub.value_counts().head(head_n).to_string())


def evaluate_model(
    name: str, model: Pipeline, X: pd.Series, y: pd.Series
) -> tuple[dict[str, float], pd.DataFrame, np.ndarray, list[str]]:
    """Cross-validated evaluation; print metrics and return rows + confusion matrix."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scoring = {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
        "f1_weighted": "f1_weighted",
    }

    scores = cross_validate(model, X, y, cv=cv, scoring=scoring)
    y_pred = cross_val_predict(model, X, y, cv=cv)

    acc_m, acc_s = scores["test_accuracy"].mean(), scores["test_accuracy"].std()
    f1mac_m, f1mac_s = scores["test_f1_macro"].mean(), scores["test_f1_macro"].std()
    f1w_m, f1w_s = scores["test_f1_weighted"].mean(), scores["test_f1_weighted"].std()

    print(f"\n{'=' * 70}")
    print(f"Model: {name}")
    print(f"{'=' * 70}")
    print(f"Accuracy:    {acc_m:.4f} ± {acc_s:.4f}")
    print(f"Macro F1:    {f1mac_m:.4f} ± {f1mac_s:.4f}")
    print(f"Weighted F1: {f1w_m:.4f} ± {f1w_s:.4f}")

    print("\nClassification Report:")
    print(classification_report(y, y_pred, digits=4, zero_division=0))

    labels = sorted(y.unique())
    cm = confusion_matrix(y, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    print("\nConfusion Matrix:")
    print(cm_df)

    summary = {
        "model": name,
        "accuracy_mean": acc_m,
        "accuracy_std": acc_s,
        "f1_macro_mean": f1mac_m,
        "f1_macro_std": f1mac_s,
        "f1_weighted_mean": f1w_m,
        "f1_weighted_std": f1w_s,
    }

    rep = classification_report(y, y_pred, digits=6, output_dict=True, zero_division=0)
    per_class_rows: list[dict[str, object]] = []
    skip = {"accuracy", "macro avg", "weighted avg"}
    for cls, m in rep.items():
        if cls in skip or not isinstance(m, dict):
            continue
        per_class_rows.append(
            {
                "model": name,
                "class": cls,
                "precision": m.get("precision", 0.0),
                "recall": m.get("recall", 0.0),
                "f1": m.get("f1-score", 0.0),
                "support": int(m.get("support", 0)),
            }
        )
    # macro / weighted as extra rows for spreadsheet convenience
    if "macro avg" in rep and isinstance(rep["macro avg"], dict):
        m = rep["macro avg"]
        per_class_rows.append(
            {
                "model": name,
                "class": "_macro_avg_",
                "precision": m.get("precision", 0.0),
                "recall": m.get("recall", 0.0),
                "f1": m.get("f1-score", 0.0),
                "support": int(m.get("support", 0)),
            }
        )
    if "weighted avg" in rep and isinstance(rep["weighted avg"], dict):
        m = rep["weighted avg"]
        per_class_rows.append(
            {
                "model": name,
                "class": "_weighted_avg_",
                "precision": m.get("precision", 0.0),
                "recall": m.get("recall", 0.0),
                "f1": m.get("f1-score", 0.0),
                "support": int(m.get("support", 0)),
            }
        )

    return summary, pd.DataFrame(per_class_rows), cm, labels


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Train/evaluate menu category classifiers from CSV.")
    p.add_argument(
        "csv",
        nargs="?",
        default="data/menu_items_train_cleaned.csv",
        help="Path to flattened menu CSV (default: data/menu_items_train_cleaned.csv)",
    )
    p.add_argument(
        "--report-base",
        default="evaluation",
        metavar="PREFIX",
        help="Filename prefix for evaluation CSVs (default: evaluation).",
    )
    p.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help=f'Directory for evaluation CSVs (default: "{PROJECT_ROOT / "results"}").',
    )
    p.add_argument(
        "--run",
        type=int,
        default=1,
        metavar="N",
        help="Run number suffix on filenames, e.g. evaluation_summary_1.csv (default: 1). Use 2, 3, … for later runs.",
    )
    p.add_argument(
        "--no-report",
        action="store_true",
        help="Do not write evaluation CSV files (terminal output only).",
    )
    p.add_argument(
        "--inspect-other",
        type=int,
        nargs="?",
        const=50,
        metavar="N",
        help='Print top N raw labels that map to "Other" (default 50), then exit.',
    )
    p.add_argument(
        "--no-confusion-heatmap",
        action="store_true",
        help="Do not save Logistic Regression confusion matrix heatmap PNG.",
    )
    args = p.parse_args()

    report_dir = resolve_report_dir(args.report_dir)
    print(f"Report directory: {report_dir}  (run suffix: {args.run})")

    csv_path = resolve_data_csv(args.csv)
    X, y, df = load_and_clean_data(str(csv_path))

    if args.inspect_other is not None:
        print(f"Total samples (after filter): {len(df)}")
        print_other_original_counts(df, head_n=int(args.inspect_other))
        return

    print(f"Total samples: {len(df)}")
    print("\nStandardized category distribution:")
    dist = df["category_standardized"].value_counts()
    print(dist)

    models = {
        "Logistic Regression": Pipeline([
            ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))
        ]),
        "Multinomial Naive Bayes": Pipeline([
            ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)),
            ("clf", MultinomialNB())
        ]),
        "Random Forest": Pipeline([
            ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)),
            ("clf", RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                class_weight="balanced_subsample"
            ))
        ]),
    }

    summary_rows: list[dict[str, object]] = []
    per_class_parts: list[pd.DataFrame] = []

    for name, model in models.items():
        srow, pc, cm, cm_labels = evaluate_model(name, model, X, y)
        summary_rows.append(srow)
        per_class_parts.append(pc)
        if (
            name == "Logistic Regression"
            and not args.no_confusion_heatmap
        ):
            out_png = report_dir / f"confusion_matrix_run{int(args.run)}.png"
            save_confusion_matrix_heatmap(cm, cm_labels, out_png)
            print(f"\n[Saved] Confusion matrix heatmap: {out_png.resolve()}")
            print_top_confusion_pairs(cm, cm_labels, k=8)
        # Save after each model so you get CSVs even if Random Forest is slow / interrupted
        if not args.no_report and args.report_base:
            sp, pp, dp = write_evaluation_csvs(
                report_dir,
                args.report_base,
                args.run,
                summary_rows,
                per_class_parts,
                dist,
            )
            print(f"\n[Saved / updated] {sp}")
            print(f"                  {pp}")
            print(f"                  {dp}")


if __name__ == "__main__":
    main()