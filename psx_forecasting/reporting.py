from __future__ import annotations

import pandas as pd

from config import PREDICTIONS_DIR, REPORTS_DIR


def generate_markdown_report() -> str:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    predictions_path = PREDICTIONS_DIR / "predictions.csv"
    report_path = REPORTS_DIR / "latest_report.md"

    if not predictions_path.exists():
        content = "# PSX Forecasting Report\n\nNo predictions have been generated yet.\n"
        report_path.write_text(content)
        return content

    predictions = pd.read_csv(predictions_path)
    latest = predictions.sort_values(["Prediction_Date", "Symbol"]).tail(20)
    evaluated = predictions.dropna(subset=["Actual_Close"])

    lines = [
        "# PSX Forecasting Report",
        "",
        "## Latest Predictions",
        "",
        latest.to_markdown(index=False),
        "",
        "## Accuracy Summary",
        "",
    ]

    if evaluated.empty:
        lines.append("No actual closing prices have been matched yet. Run the update after the next trading day.")
    else:
        lines.extend(
            [
                f"Mean absolute error: {evaluated['Error'].abs().mean():.2f}",
                f"Mean percentage error: {evaluated['Error_Pct'].mean():.2f}%",
            ]
        )

    content = "\n".join(lines) + "\n"
    report_path.write_text(content)
    return content
