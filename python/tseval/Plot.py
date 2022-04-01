import itertools
from pathlib import Path
from typing import *

import matplotlib as mpl
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from seutil import IOUtils, latex, LoggingUtils

from tseval.Environment import Environment
from tseval.Macros import Macros


class Plot:

    logger = LoggingUtils.get_logger(__name__, LoggingUtils.DEBUG if Environment.is_debug else LoggingUtils.INFO)

    @classmethod
    def init_plot_libs(cls):
        # Initialize seaborn
        sns.set()
        sns.set_palette("Dark2")
        sns.set_context("paper")
        # Set matplotlib fonts
        mpl.rcParams["axes.titlesize"] = 24
        mpl.rcParams["axes.labelsize"] = 24
        mpl.rcParams["font.size"] = 18
        mpl.rcParams["xtick.labelsize"] = 24
        mpl.rcParams["xtick.major.size"] = 14
        mpl.rcParams["xtick.minor.size"] = 14
        mpl.rcParams["ytick.labelsize"] = 24
        mpl.rcParams["ytick.major.size"] = 14
        mpl.rcParams["ytick.minor.size"] = 14
        mpl.rcParams["legend.fontsize"] = 18
        mpl.rcParams["legend.title_fontsize"] = 18
        # print(mpl.rcParams)

    def __init__(self):
        self.plots_dir: Path = Macros.paper_dir / "figs"
        IOUtils.mk_dir(self.plots_dir)
        self.init_plot_libs()

    def make_plots(self, options: dict):
        which = options.pop("which")

        if which == "dataset-metrics-dist":
            self.dataset_metrics_dist(
                task=options["task"],
            )
        else:
            self.logger.warning(f"Unknown plot {which}")

    def dataset_metrics_dist(self, task: str):
        plots_sub_dir = self.plots_dir / f"dataset-{task}"
        plots_sub_dir_rel = str(plots_sub_dir.relative_to(Macros.paper_dir))
        IOUtils.rm_dir(plots_sub_dir)
        plots_sub_dir.mkdir(parents=True)

        # Load metrics list into a DataFrame
        label_x = "code"
        max_x = 200
        if task == "CG":
            label_y = "comment"
            max_y = 60
        else:
            label_y = "name"
            max_y = 8
        lod: List[dict] = []
        seen_split_combination = set()
        for setup in Macros.split_types:
            metrics_list = IOUtils.load(Macros.results_dir / "metrics" / f"setup-dataset-metrics-list_{task}_{setup}.pkl", IOUtils.Format.pkl)
            for sn in [Macros.train, Macros.val, Macros.test_standard]:
                for i, (x, y) in enumerate(zip(
                        metrics_list[f"{sn}_len-{label_x}"], metrics_list[f"{sn}_len-{label_y}"],
                )):
                    lod.append({
                        "i": i,
                        "set_name": f"{sn}-{setup}",
                        label_x: x,
                        label_y: y,
                    })

            for s1, s2 in Macros.get_pairwise_split_types_with(setup):
                if (s1, s2) in seen_split_combination:
                    continue
                for i, (x, y) in enumerate(zip(
                        metrics_list[f"{Macros.test_common}-{s1}-{s2}_len-{label_x}"],
                        metrics_list[f"{Macros.test_common}-{s1}-{s2}_len-{label_y}"],
                )):
                    lod.append({
                        "i": i,
                        "set_name": f"{Macros.test_common}-{s1}-{s2}",
                        label_x: x,
                        label_y: y,
                    })
                seen_split_combination.add((s1, s2))

        df = pd.DataFrame(lod)

        # Make plots
        for sn, df_sn in df.groupby("set_name", as_index=False):
            sn: str
            if sn in [f"{x}-{Macros.temporally}" for x in [Macros.train, Macros.val, Macros.test_standard]] + [f"{Macros.test_common}-{Macros.cross_prj}-{Macros.temporally}"]:
                display_xlabel = "len(" + label_x + ")"
            else:
                display_xlabel = None
            if sn.startswith(Macros.train):
                display_ylabel = "len(" + label_y + ")"
            else:
                display_ylabel = None

            fig = sns.jointplot(
                data=df_sn,
                x=label_x, y=label_y,
                kind="hist",
                xlim=(0, max_x),
                ylim=(0, max_y),
                height=6,
                ratio=3,
                space=.01,
                joint_kws=dict(
                    bins=(12, min(12, max_y-1)),
                    binrange=((0, max_x), (0, max_y)),
                    pmax=.5,
                ),
                color="royalblue",
            )
            fig.set_axis_labels(display_xlabel, display_ylabel)
            plt.tight_layout()
            fig.savefig(plots_sub_dir / f"{sn}.pdf")

        # Generate a tex file that organizes the plots
        f = latex.File(plots_sub_dir / f"plot.tex")
        f.append(r"\begin{center}")
        f.append(r"\begin{footnotesize}")
        f.append(r"\begin{tabular}{|l|c|c|c || l|c|}")
        f.append(r"\hline")
        f.append(r"& \textbf{ATrain} & \textbf{\AVal} & \textbf{\ATestS} & & \textbf{\ATestC} \\")

        for sn_l, (s1_r, s2_r) in zip(Macros.split_types, itertools.combinations(Macros.split_types, 2)):
            f.append(r"\hline")
            f.append(latex.Macro(f"TH-ds-{sn_l}").use())
            for sn in [Macros.train, Macros.val, Macros.test_standard]:
                f.append(r" & \begin{minipage}{.18\textwidth}\includegraphics[width=\textwidth]{"
                         + f"{plots_sub_dir_rel}/{sn}-{sn_l}"
                         + r"}\end{minipage}")
            f.append(" & " + latex.Macro(f"TH-ds-{s1_r}-{s2_r}").use())
            f.append(r" & \begin{minipage}{.18\textwidth}\includegraphics[width=\textwidth]{"
                     + f"{plots_sub_dir_rel}/{Macros.test_common}-{s1_r}-{s2_r}"
                     + r"}\end{minipage}")
            f.append(r"\\")
        f.append(r"\hline")
        f.append(r"\end{tabular}")
        f.append(r"\end{footnotesize}")
        f.append(r"\end{center}")
        f.save()
