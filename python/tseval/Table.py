import collections
import itertools
from pathlib import Path

from seutil import IOUtils, latex, LoggingUtils
from seutil.latex.Macro import Macro

from tseval.Environment import Environment
from tseval.Macros import Macros


class Table:
    logger = LoggingUtils.get_logger(__name__, LoggingUtils.DEBUG if Environment.is_debug else LoggingUtils.INFO)

    COLSEP = "COLSEP"
    ROWSEP = "ROWSEP"

    def __init__(self):
        self.tables_dir: Path = Macros.paper_dir / "tables"
        IOUtils.mk_dir(self.tables_dir)

        self.metrics_dir: Path = Macros.results_dir / "metrics"
        return

    def make_tables(self, options):
        which = options.pop("which")
        if which == "dataset-metrics":
            self.make_numbers_dataset_metrics()
            self.make_table_dataset_metrics_small()
            for task in [Macros.com_gen, Macros.met_nam]:
                self.make_table_dataset_metrics(task)
        else:
            self.logger.warning(f"Unknown table name {which}")

    def make_numbers_dataset_metrics(self):
        file = latex.File(self.tables_dir / f"numbers-dataset-metrics.tex")

        dataset_filtered_metrics = IOUtils.load(Macros.results_dir / "metrics" / f"raw-data-filtered.json", IOUtils.Format.json)
        for k, v in dataset_filtered_metrics.items():
            file.append_macro(latex.Macro(f"ds-filter-{k}", f"{v:,d}"))

        dataset_metrics = IOUtils.load(Macros.results_dir / "metrics" / f"split-dataset-metrics_Full.json", IOUtils.Format.json)
        for k, v in dataset_metrics.items():
            fmt = f",d" if type(v) == int else f",.2f"
            file.append_macro(latex.Macro(f"ds-{k}", f"{v:{fmt}}"))

        for task in [Macros.com_gen, Macros.met_nam]:
            for split in Macros.split_types:
                setup_metrics = IOUtils.load(Macros.results_dir / "metrics" / f"setup-dataset-metrics_{task}_{split}.json", IOUtils.Format.json)
                for k, v in setup_metrics.items():
                    fmt = f",d" if type(v) == int else f",.2f"
                    if k.startswith("all_"):
                        # Skip all_ metrics
                        continue
                    else:
                        # Replace train/val/test_standard to train-split/val-split/test_standard-split
                        for x in [Macros.train, Macros.val, Macros.test_standard]:
                            if k.startswith(f"{x}_"):
                                k = f"{x}-{split}_" + k[len(f"{x}_"):]
                                break
                        if f"ds-{task}-{k}" in file.macros_indexed:
                            continue
                        file.append_macro(latex.Macro(f"ds-{task}-{k}", f"{v:{fmt}}"))

        file.save()

    def make_table_dataset_metrics(self, task: str):
        f = latex.File(self.tables_dir / f"table-dataset-metrics-{task}.tex")

        metric_2_th = collections.OrderedDict()
        # metric_2_th["num-proj"] = r"\multicolumn{2}{c|}{\UseMacro{TH-ds-num-project}}"
        metric_2_th["num-data"] = r"\multicolumn{2}{c|}{\UseMacro{TH-ds-num-data}}"
        metric_2_th["sep-data"] = self.ROWSEP
        metric_2_th["len-code-AVG"] = r"& \UseMacro{TH-ds-len-code-avg}"
        # metric_2_th["len-code-MODE"] = r"& \UseMacro{TH-ds-len-code-mode}"
        # metric_2_th["len-code-MEDIAN"] = r"& \UseMacro{TH-ds-len-code-median}"
        metric_2_th["len-code-le-100"] = r"& \UseMacro{TH-ds-len-code-le100}"
        metric_2_th["len-code-le-150"] = r"& \UseMacro{TH-ds-len-code-le150}"
        metric_2_th["len-code-le-200"] = r"\multirow{-4}{*}{\UseMacro{TH-ds-len-code}} & \UseMacro{TH-ds-len-code-le200}"
        if task == Macros.com_gen:
            metric_2_th["sep-cg"] = self.ROWSEP
            metric_2_th["len-comment-AVG"] = r"& \UseMacro{TH-ds-len-comment-avg}"
            # metric_2_th["len-comment-MODE"] = r"& \UseMacro{TH-ds-len-comment-mode}"
            # metric_2_th["len-comment-MEDIAN"] = r"& \UseMacro{TH-ds-len-comment-median}"
            metric_2_th["len-comment-le-20"] = r"& \UseMacro{TH-ds-len-comment-le20}"
            metric_2_th["len-comment-le-30"] = r"& \UseMacro{TH-ds-len-comment-le30}"
            metric_2_th["len-comment-le-50"] = r"\multirow{-4}{*}{\UseMacro{TH-ds-len-comment}} & \UseMacro{TH-ds-len-comment-le50}"
        if task == Macros.met_nam:
            metric_2_th["sep-mn"] = self.ROWSEP
            metric_2_th["len-name-AVG"] = r"& \UseMacro{TH-ds-len-name-avg}"
            # metric_2_th["len-name-MODE"] = r"& \UseMacro{TH-ds-len-name-mode}"
            # metric_2_th["len-name-MEDIAN"] = r"& \UseMacro{TH-ds-len-name-median}"
            metric_2_th["len-name-le-2"] = r"& \UseMacro{TH-ds-len-name-le2}"
            metric_2_th["len-name-le-3"] = r"& \UseMacro{TH-ds-len-name-le3}"
            metric_2_th["len-name-le-6"] = r"\multirow{-4}{*}{\UseMacro{TH-ds-len-name}} & \UseMacro{TH-ds-len-name-le6}"

        cols = sum(
            [
                [f"{s1}-{s2}" for s1 in [Macros.train, Macros.val, Macros.test_standard]] + [self.COLSEP]
                for s2 in [Macros.mixed_prj, Macros.cross_prj, Macros.temporally]
            ]
            , [],
        ) + [f"{Macros.test_common}-{x}-{y}"
             for x, y in itertools.combinations([Macros.mixed_prj, Macros.cross_prj, Macros.temporally], 2)]

        # Header
        f.append(r"\begin{table*}[t]")
        f.append(r"\begin{footnotesize}")
        f.append(r"\begin{center}")
        table_name = f"dataset-metrics-{task}"

        f.append(
            r"\begin{tabular}{ l@{\hspace{2pt}}c@{\hspace{2pt}} | "
            r"r@{\hspace{4pt}}r@{\hspace{4pt}}r @{\hspace{3pt}}c@{\hspace{3pt}} "
            r"r@{\hspace{4pt}}r@{\hspace{4pt}}r @{\hspace{3pt}}c@{\hspace{3pt}} "
            r"r@{\hspace{4pt}}r@{\hspace{4pt}}r @{\hspace{3pt}}c@{\hspace{3pt}} "
            r"r@{\hspace{4pt}}r@{\hspace{4pt}}r }"
        )

        f.append(r"\toprule")

        # Line 1
        f.append(
            r"\multicolumn{2}{c|}{}"
            r" & \multicolumn{3}{c}{\UseMacro{TH-ds-MP}} &"
            r" & \multicolumn{3}{c}{\UseMacro{TH-ds-CP}} &"
            r" & \multicolumn{3}{c}{\UseMacro{TH-ds-T}} &"
            r" & \UseMacro{TH-ds-MP-CP} & \UseMacro{TH-ds-MP-T} & \UseMacro{TH-ds-CP-T}"
            r" \\\cline{3-5}\cline{7-9}\cline{11-13}\cline{15-17}"
        )

        # Line 2
        f.append(
            r"\multicolumn{2}{c|}{\multirow{-2}{*}{\THDSStat}}"
            r" & \UseMacro{TH-ds-train} & \UseMacro{TH-ds-val} & \UseMacro{TH-ds-test_standard} &"
            r" & \UseMacro{TH-ds-train} & \UseMacro{TH-ds-val} & \UseMacro{TH-ds-test_standard} &"
            r" & \UseMacro{TH-ds-train} & \UseMacro{TH-ds-val} & \UseMacro{TH-ds-test_standard} &"
            r" & \multicolumn{3}{c}{\UseMacro{TH-ds-test_common}} \\"
        )

        f.append(r"\midrule")

        for metric, row_th in metric_2_th.items():
            if row_th == self.ROWSEP:
                f.append(r"\midrule")
                continue

            f.append(row_th)

            for col in cols:
                if col == self.COLSEP:
                    f.append(" & ")
                    continue
                f.append(" & " + latex.Macro(f"ds-{task}-{col}_{metric}").use())

            f.append(r"\\")

        # Footer
        f.append(r"\bottomrule")
        f.append(r"\end{tabular}")
        f.append(r"\end{center}")
        f.append(r"\end{footnotesize}")
        f.append(r"\vspace{" + latex.Macro(f"TV-{table_name}").use() + "}")
        f.append(r"\caption{" + latex.Macro(f"TC-{table_name}").use() + r"}")
        f.append(r"\end{table*}")

        f.save()

    def make_table_dataset_metrics_small(self):
        f = latex.File(self.tables_dir / f"table-dataset-metrics-small.tex")

        # Header
        f.append(r"\begin{table}[t]")
        f.append(r"\begin{footnotesize}")
        f.append(r"\begin{center}")
        table_name = f"dataset-metrics-small"

        f.append(r"\begin{tabular}{ @{} l | c @{\hspace{5pt}} r @{\hspace{5pt}} r @{\hspace{5pt}} r @{\hspace{3pt}}c@{\hspace{3pt}} c @{\hspace{5pt}} r@{} }")
        f.append(r"\toprule")
        f.append(r"\textbf{Task} & & \textbf{\ATrain} & \textbf{\AVal} & \textbf{\ATestS} & & & \textbf{\ATestC} \\")

        for task in [Macros.com_gen, Macros.met_nam]:

            f.append(r"\midrule")
            
            for i, (m, p) in enumerate(zip(
                [Macros.mixed_prj, Macros.cross_prj, Macros.temporally], 
                [f"{x}-{y}" for x, y in itertools.combinations([Macros.mixed_prj, Macros.cross_prj, Macros.temporally], 2)],
            )):
                if i == 2:
                    f.append(r"\multirow{-3}{*}{\rotatebox[origin=c]{90}{" + latex.Macro(f"TaskM_{task}").use() + r"}}")

                f.append(" & " + latex.Macro(f"TH-ds-{m}").use())
                for sn in [Macros.train, Macros.val, Macros.test_standard]:
                    f.append(" & " + latex.Macro(f"ds-{task}-{sn}-{m}_num-data").use())
                f.append(r" & \tikz[remember picture, baseline] \node[inner sep=2pt, outer sep=0, yshift=1ex] (" + task + m + r"-base) {\phantom{XX}};")
                f.append(" & " + latex.Macro(f"TH-ds-{p}").use())
                f.append(" & " + latex.Macro(f"ds-{task}-{Macros.test_common}-{p}_num-data").use())
                f.append(r"\\")

        f.append(r"\bottomrule")
        f.append(r"\end{tabular}")

        f.append(r"\begin{tikzpicture}[remember picture, overlay, thick]")
        for task in [Macros.com_gen, Macros.met_nam]:
            for r, (l1, l2) in zip(
                [Macros.mixed_prj, Macros.cross_prj, Macros.temporally],
                [
                    (Macros.mixed_prj, Macros.cross_prj),
                    (Macros.mixed_prj, Macros.temporally),
                    (Macros.cross_prj, Macros.temporally),
                ]
            ):
                f.append(r"\draw[->] (" + task + l1 + r"-base.west) .. controls ($(" + task + r + r"-base.east) - (1em,0)$) .. (" + task + r + r"-base.east);")
                f.append(r"\draw (" + task + l2 + r"-base.west) .. controls ($(" + task + r + r"-base.east) - (1em,0)$) .. (" + task + r + r"-base.east);")

        f.append(r"\end{tikzpicture}")

        f.append(r"\end{center}")
        f.append(r"\end{footnotesize}")
        f.append(r"\vspace{" + latex.Macro(f"TV-{table_name}").use() + "}")
        f.append(r"\caption{" + latex.Macro(f"TC-{table_name}").use() + "}")
        f.append(r"\end{table}")

        f.save()
