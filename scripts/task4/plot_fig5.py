#!/usr/bin/env python
"""Draw the two Fig.5 line charts from a Source Data Excel workbook.

Read the five trials in the Top1 and map sheets without changing their values.
Each model is shown as a line connecting its five trial scores.
"""

from __future__ import annotations

import argparse
import os
from numbers import Real
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


MODELS = ("NeuroSTORM", "SwiFT", "TFF", "BrainLM", "Com-brainTF", "LG-GNN", "BNT", "BrainGNN")
COLORS = ("#B64342", "#315E91", "#62809A", "#796391",
          "#7A9788", "#B38B58", "#666666", "#6A9B9E")
MARKERS = ("o", "s", "^", "D", "v", "P", "X", ">")
LINE_STYLES = ("-", "-", "--", "-.", (0, (4, 2)),
               (0, (2, 1)), ":", (0, (6, 2, 1, 2)))


def load_source_data(source_path):
    """Read the supplied five rows verbatim; each cell is one plotted value."""
    import openpyxl

    panels = []
    book = openpyxl.load_workbook(source_path, read_only=True, data_only=True)
    try:
        for sheet_name, metric in (("Top1", "top1"), ("map", "mAP")):
            if sheet_name not in book.sheetnames:
                raise ValueError(f"Source Data is missing sheet {sheet_name!r}")
            sheet = book[sheet_name]
            if (sheet.max_row, sheet.max_column) != (6, 9):
                raise ValueError(f"{sheet_name}: expected five trials and eight model columns")
            models = tuple(sheet.cell(1, c).value for c in range(2, 10))
            if len(set(models)) != len(MODELS) or set(models) != set(MODELS):
                raise ValueError(f"{sheet_name}: expected the eight ReID model names")
            values = np.empty((5, len(MODELS)), dtype=float)
            for trial in range(5):
                for index, model in enumerate(MODELS):
                    cell = sheet.cell(trial + 2, models.index(model) + 2)
                    number = cell.value
                    if (not isinstance(number, Real) or isinstance(number, bool)
                            or not np.isfinite(number) or not 0 <= number <= 100):
                        raise ValueError(f"{sheet_name}!{cell.coordinate}: expected a percentage from 0 to 100")
                    values[trial, index] = number
            panels.append(dict(sheet=sheet_name, metric=metric, models=MODELS, values=values))
    finally:
        book.close()
    return panels


def save_figure(fig, output_dir):
    for extension in ("png", "svg", "pdf"):
        fig.savefig(output_dir / f"Fig5_lineplot.{extension}", dpi=600 if extension == "png" else None,
                    facecolor="white", metadata={"Creator": "plot_fig5.py"} if extension == "pdf" else None)
    plt.close(fig)


def draw(panels, output_dir):
    """Connect the five Source Data trial scores using the reference styling."""
    with plt.rc_context({"font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
                         "text.color": "black", "axes.labelcolor": "black",
                         "xtick.color": "black", "ytick.color": "black"}):
        fig, axes = plt.subplots(1, 2, figsize=(183 / 25.4, 76 / 25.4), sharey=True)
        fig.subplots_adjust(left=0.082, right=0.98, bottom=0.18, top=0.86, wspace=0.26)
        x = np.arange(1, 6)
        series = []
        available_fonts = {font.name for font in font_manager.fontManager.ttflist}
        panel_font = next(name for name in ("Arial Black", "Arial", "DejaVu Sans")
                          if name in available_fonts)
        for panel_id, (ax, panel) in enumerate(zip(axes, panels)):
            lines = []
            for index, model in enumerate(MODELS):
                line, = ax.plot(
                    x, panel["values"][:, index], label=model,
                    color=COLORS[index], marker=MARKERS[index],
                    linestyle=LINE_STYLES[index],
                    linewidth=1.7 if index == 0 else 1.0,
                    markersize=4.2 if index == 0 else 3.5,
                    markerfacecolor=COLORS[index] if index == 0 else "white",
                    markeredgewidth=0.9, zorder=5 if index == 0 else 3,
                )
                line.set_gid(f"source-series-{panel['metric']}-{model.lower().replace('-', '')}")
                lines.append(line)
            series.append(lines)
            ax.text(-0.17, 1.035, "ab"[panel_id], transform=ax.transAxes,
                    fontfamily=panel_font, fontsize=9.3,
                    fontweight="black", va="bottom")
            ax.set_xlim(0.8, 5.2)
            ax.set_ylim(75, 100)
            ax.set_xticks(x, [f"Trial {i}" for i in x])
            ax.set_yticks(np.arange(75, 101, 5))
            ax.set_xlabel("HCP-YA (n = 216)", labelpad=6, fontsize=8)
            label = "Top-1 accuracy (%)" if panel_id == 0 else "Mean average precision (%)"
            ax.set_ylabel(label, labelpad=7, fontsize=8)
            ax.grid(axis="y", color="#E5E5E5", linewidth=0.25, zorder=0)
            ax.tick_params(axis="both", length=2.7, width=0.25, pad=3, labelsize=7)
            ax.tick_params(axis="y", labelleft=True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color("black")
                ax.spines[side].set_linewidth(0.25)
        fig.legend(series[0], MODELS, loc="upper center", bbox_to_anchor=(0.53, 0.98), ncol=8,
                   columnspacing=1.0, handlelength=2.2, handletextpad=0.4,
                   labelspacing=0.7, borderaxespad=0, fontsize=7, frameon=False)
        save_figure(fig, output_dir)


def register_paper_fonts():
    system_fonts = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    for name in ("arial.ttf", "arialbd.ttf", "ariali.ttf", "arialbi.ttf", "ariblk.ttf"):
        if (system_fonts / name).is_file():
            font_manager.fontManager.addfont(system_fonts / name)
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "svg.fonttype": "path", "axes.unicode_minus": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="Source Data Excel workbook containing the Top1 and map sheets")
    parser.add_argument("--output-dir", type=Path, default=Path("results/fig5"))
    args = parser.parse_args()
    data = load_source_data(args.source)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    register_paper_fonts()
    draw(data, args.output_dir)
    for index, model in enumerate(MODELS):
        top1 = data[0]["values"][:, index].mean()
        ap = data[1]["values"][:, index].mean()
        print(f"{model}: Source Data mean Top-1 {top1:.6f}%, mAP {ap:.6f}%")
    print(f"Figure saved in {args.output_dir}")


if __name__ == "__main__":
    main()
