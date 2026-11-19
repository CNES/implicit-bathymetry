#!/usr/bin/env python3
# Copyright (c) 2025 GeoUtils developers
#
# This file is part of the implicit_bathy project:
#  https://.../implicit_bathy
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Analyze report
"""

import argparse
import glob
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def color_specific(val):
    if val in (0, 1):
        return "background-color: red; color: white"
    if 2 <= val <= 10:
        return "background-color: green; color: white"
    return ""


def analyze_csv(output_path):
    """
    Analyse report.csv
    """
    all_stats_data = []
    for report_file in sorted(Path(output_path).glob("out_path*/report.csv")):
        stats_data = {}
        with open(report_file, encoding="utf-8") as f:
            for stats in f:
                if "," in stats:
                    _, col_stats, val_stats = stats.strip().split(",", 2)
                    stats_data[col_stats.strip()] = val_stats.strip()
        stats_data["name"] = Path(report_file).parent.name
        all_stats_data.append(stats_data)

    out_synthesis = Path(output_path) / "synthesis" / "global_report.csv"

    df_stats = pd.DataFrame(all_stats_data)
    df_stats.to_csv(out_synthesis)
    df_stats = df_stats[["name"] + [c for c in df_stats.columns if c != "name"]]
    df_stats = df_stats.dropna(axis=1, how="all")

    colonnes_str = ["name", "water_losses"]
    indicators = df_stats.columns.difference(colonnes_str)
    df_stats[indicators] = df_stats[indicators].apply(
        pd.to_numeric, errors="coerce"
    )
    cols = list(df_stats.columns)
    colonnes_stats = [
        col for col in cols if col.startswith(("E_", "mre", "me"))
    ]
    colonnes_rmse = [
        col for col in cols if col.startswith(("mare", "q68", "mre", "mae"))
    ]

    col_ratio = [
        c for c in df_stats.columns if c.startswith("water_ratio_epoch")
    ]
    colors = ["red", "yellow", "green", "yellow", "red"]

    custom_cmap = mcolors.LinearSegmentedColormap.from_list(
        "custom_red_green_red", colors
    )

    styled_df = (
        df_stats.style.applymap(
            color_specific,
            subset=[
                col_ratio[0],
                "nb_points_pw_010",
                "nb_points_pw_099",
                "nb_points_pw_095",
            ],
        )
        .background_gradient(
            custom_cmap, subset=colonnes_stats, vmin=-0.5, vmax=0.5
        )
        .background_gradient(subset=colonnes_rmse, cmap="RdYlGn_r")
        .background_gradient(subset=["nll_losses"], cmap="RdYlGn_r")
    )

    excel_file = Path(output_path) / "synthesis" / "global_report.xlsx"
    with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
        styled_df.to_excel(writer, index=False, sheet_name="Feuille1")


def synthesis_png(output_path):
    """
    Compilation png results
    """
    out_synthesis = Path(output_path) / "synthesis"
    folder_inference = sorted(Path(output_path).glob("out_path*/inference"))

    file_png = set()
    for im_inference in folder_inference:
        file_png.update(
            [
                p.name
                for p in Path(im_inference).iterdir()
                if p.suffix.lower() == ".png"
            ]
        )

    for file_im in file_png:
        images = []
        titles = []
        for file_inf in folder_inference:
            path = Path(file_inf) / file_im
            if Path(path).exists():
                img = mpimg.imread(Path(path))
                images.append(img)
                titles.append(Path(file_inf).parent.name)
        if not images:
            continue
        n = len(images)
        max_col = 4
        n_rows = math.ceil(n / max_col)
        n_cols = min(n, max_col)
        _, axes = plt.subplots(
            n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), dpi=300
        )
        axes = axes.flatten()
        if n == 1:
            axes = [axes]
        for ax, img, title in zip(axes, images, titles):
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(title, fontsize=8)

        for ax in axes[len(images) :]:
            ax.axis("off")
            ax.set_title("")
        plt.tight_layout()
        plt.savefig(Path(out_synthesis) / file_im)
        plt.close()


def compute_boxplot(output_path, csv_file):
    """Compute Boxplot"""

    df_csv = pd.read_csv(csv_file)
    param_csv = df_csv.columns[0]

    if "seed" in df_csv[param_csv].astype(str).values:
        files = sorted(
            glob.glob(os.path.join(output_path, "synthesis", "*.xlsx"))
        )
        stats = {}
        for f in files:
            df = pd.read_excel(f, index_col="name")
            df_numeric = df.select_dtypes(include="number")
            folder_name = os.path.basename(os.path.dirname(os.path.dirname(f)))
            for col in df_numeric.columns:
                if col not in stats:
                    stats[col] = {"folders": [], "data": [], "seed_folders": []}
                stats[col]["folders"].append(folder_name)
                stats[col]["seed_folders"].append(
                    df_numeric[col].index.tolist()
                )
                stats[col]["data"].append(df_numeric[col].dropna().values)

        pred_list = list(stats.keys())
        groups = defaultdict(list)

        for item in pred_list:
            if item.endswith("hc") or item.endswith("pred"):
                prefix = item.rsplit("_", 1)[0]
                groups[prefix].append(item)

        for prefix, items in groups.items():
            if str(prefix).startswith(("mae_height", "mare_area", "maern")):
                if f"{prefix}_hc" in items and f"{prefix}_pred" in items:
                    seen_labels = []
                    pred_col = items[1]
                    hc_col = items[0]

                    folders = stats[pred_col]["folders"]
                    x = np.arange(len(folders))

                    plt.figure(figsize=(16, 8))

                    plt.boxplot(
                        stats[pred_col]["data"],
                        positions=x,
                        widths=0.5,
                        showfliers=False,
                    )

                    for i, (d, names) in enumerate(
                        zip(
                            stats[pred_col]["data"],
                            stats[pred_col]["seed_folders"],
                        )
                    ):
                        jitter = np.random.normal(0, 0.05, size=len(d))
                        for j, (val, name) in enumerate(zip(d, names)):
                            xpos = x[i] + jitter[j]
                            if (
                                name not in seen_labels
                                and len(seen_labels) < 10
                            ):
                                label = name.split("_")[2]
                                seen_labels.append(label)
                            else:
                                label = None

                            plt.scatter(xpos, val, alpha=0.7, s=30, label=label)

                    means_hc = [np.mean(d) for d in stats[hc_col]["data"]]
                    stds_hc = [np.std(d, ddof=1) for d in stats[hc_col]["data"]]
                    plt.errorbar(
                        x,
                        means_hc,
                        stds_hc,
                        fmt="s",
                        capsize=5,
                        color="tab:red",
                        label=hc_col,
                    )
                    plt.xticks([])
                    plt.xlabel(f"{prefix} values for different seeds")
                    plt.ylabel(str(prefix))
                    plt.title(f"{prefix} distribution for different seeds")

                    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
                    plt.tight_layout()

                    plt.savefig(
                        os.path.join(
                            output_path, "synthesis", f"{pred_col}_boxplot.png"
                        ),
                        dpi=300,
                    )
                    plt.close()


def main():
    """Main"""
    parser = argparse.ArgumentParser(description="CLI to analyze report")
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to output path with reports",
        required=True,
    )
    parser.add_argument(
        "--csv_file", type=str, help="Path to csv_file", required=True
    )
    args = parser.parse_args()
    output_path = args.output_path
    csv_file = args.csv_file
    analyze_csv(output_path)
    synthesis_png(output_path)
    compute_boxplot(output_path, csv_file)


if __name__ == "__main__":
    sys.exit(main())
