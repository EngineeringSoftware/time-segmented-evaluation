import collections
import itertools
import random
import stat
import tempfile
from pathlib import Path
from typing import *

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from recordclass import RecordClass
from seutil import IOUtils, latex, LoggingUtils
from tqdm import tqdm

from tseval.data.MethodData import MethodData
from tseval.Environment import Environment
from tseval.eval.EvalMetrics import EvalMetrics
from tseval.Macros import Macros
from tseval.util.ModelUtils import ModelUtils
from tseval.Utils import Utils

logger = LoggingUtils.get_logger(__name__)


class ModelSpec(RecordClass):
    name: str = None
    exps: List[str] = None


class ExperimentsSpec(RecordClass):
    task: str = None
    setups: List[str] = None
    num_trials: int = 3
    metrics: List[str] = None
    models: List[ModelSpec] = None
    table_args: dict = None
    plot_args: dict = None


def bootstrap(
        full_a_scores: List[float],
        full_b_scores: List[float],
        num_samples: int = 10_000,
        test_size: int = 1000,
        is_pairwise: bool = True,
        batch_size: int = 10_000,
) -> float:
    """
    Uses bootstrap method to perform the statistical significance test for the hypothesis: a > b, and returns the
    probability (p-value) to reject the hypothesis.

    Args:
        full_a_scores: the full set of a values to sample from
        full_b_scores: the full set of b values to sample from
        num_samples: the number of samples to repeat
        test_size: the size at each sample
        is_pairwise: if the test is pairwise (requires len(full_a_scores) == len(full_b_scores))
    Returns:
        the p-value, which should be smaller than or equal to the significance level (usually 5%) to claim
        that a > b is a statistical significant result
    """
    if is_pairwise and len(full_a_scores) != len(full_b_scores):
        LoggingUtils.log_and_raise(logger,
            "Cannot perform pairwise significance test if two sets do not have the same size.",
            Exception)

    a_scores = []
    b_scores = []

    # Prepare torch device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    tensor_full_a = torch.tensor(full_a_scores, dtype=torch.float, device=device)
    tensor_full_b = torch.tensor(full_b_scores, dtype=torch.float, device=device)
    for sample_i in range(0, num_samples, batch_size):
        if sample_i + batch_size > num_samples:
            batch = num_samples - sample_i
        else:
            batch = batch_size

        if is_pairwise:
            sample_indices = torch.multinomial(
                torch.ones((batch, len(full_a_scores)), dtype=torch.float, device=device),
                test_size,
                replacement=True,
            )
            tensor_a = torch.gather(tensor_full_a.repeat(batch, 1), 1, sample_indices)
            tensor_b = torch.gather(tensor_full_b.repeat(batch, 1), 1, sample_indices)
            del sample_indices
        else:
            sample_indices_a = torch.multinomial(
                torch.ones((batch, len(full_a_scores)), dtype=torch.float, device=device),
                test_size,
                replacement=True,
            )
            tensor_a = torch.gather(tensor_full_a.repeat(batch, 1), 1, sample_indices_a)
            sample_indices_b = torch.multinomial(
                torch.ones((batch, len(full_b_scores)), dtype=torch.float, device=device),
                test_size,
                replacement=True,
            )
            tensor_b = torch.gather(tensor_full_b.repeat(batch, 1), 1, sample_indices_b)
            del sample_indices_a, sample_indices_b

        a_scores.extend(tensor_a.mean(dim=1).tolist())
        b_scores.extend(tensor_b.mean(dim=1).tolist())
        del tensor_a, tensor_b

    significance_a_over_b = 0
    for i in range(num_samples):
        if a_scores[i] > b_scores[i]:
            significance_a_over_b += 1

    return 1 - significance_a_over_b / float(num_samples)


class ExperimentsAnalyzer:

    def __init__(self, exps_spec_path: Path, output_prefix: str = None):
        if output_prefix is None:
            output_prefix = exps_spec_path.stem

        self.exps_spec_path = exps_spec_path
        self.spec: ExperimentsSpec = IOUtils.dejsonfy(IOUtils.load(exps_spec_path, IOUtils.Format.yaml), ExperimentsSpec)
        self.output_dir = Macros.results_dir / "exps" / output_prefix
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.table_dir = Macros.paper_dir / "tables"
        self.table_dir.mkdir(parents=True, exist_ok=True)
        self.plot_sub_dir = Macros.paper_dir / "figs" / f"results-{self.spec.task}"

    def check_files(self):
        errors = collections.defaultdict(list)
        warnings = collections.defaultdict(list)
        dvc_pull = []
        for model in self.spec.models:
            if len(model.exps) < self.spec.num_trials:
                errors["not_enough_trials"].append(f"{model}: {len(model.exps)}/{self.spec.num_trials}")
            elif len(model.exps) > self.spec.num_trials:
                warnings["too_many_trials"].append(f"{model}: {len(model.exps)}/{self.spec.num_trials}")
            for setup in self.spec.setups:
                for exp in model.exps:
                    # Check existence of result/metric directories, and suggest dvc pull
                    result_dir = Macros.work_dir / self.spec.task / "result" / setup / exp
                    result_dir_dvc = Macros.work_dir / self.spec.task / "result" / setup / f"{exp}.dvc"
                    if not result_dir.is_dir():
                        if not result_dir_dvc.is_file():
                            errors["missing_result"].append(f"{setup}, {model}, {exp}: {result_dir}")
                        else:
                            dvc_pull.append(result_dir_dvc)
                    else:
                        dvc_pull.append(result_dir_dvc)

                    metric_dir = Macros.work_dir / self.spec.task / "metric" / setup / exp
                    metric_dir_dvc = Macros.work_dir / self.spec.task / "metric" / setup / f"{exp}.dvc"
                    if not metric_dir.is_dir():
                        if not metric_dir_dvc.is_file():
                            errors["missing_metric"].append(f"{setup}, {model}, {exp}: {metric_dir}")
                        else:
                            dvc_pull.append(metric_dir_dvc)
                    else:
                        dvc_pull.append(metric_dir_dvc)

                    if result_dir.is_dir() and metric_dir.is_dir():
                        # Check completeness of result/metric files
                        sns = self.get_result_prefixes(setup)
                        result_suffixes = ["golds.jsonl", "predictions.jsonl"]
                        metric_suffixes = ["metrics.json", "metrics.txt", "metrics_list.pkl"]
                        for sns, action in [
                            ([Macros.val], Macros.val),
                            ([Macros.test_standard], Macros.test_standard),
                            ([f"{Macros.test_common}-{x}-{y}" for x, y in Macros.get_pairwise_split_types_with(setup)],
                             Macros.test_common),
                        ]:
                            if any(not (result_dir / f"{sn}_{sf}").is_file() for sf in result_suffixes for sn in sns):
                                errors["missing_file"].append(
                                    f"python -m tseval.main exp_eval --task={self.spec.task} --setup_name={setup} --exp_name={exp} --action={action} && python -m tseval.main exp_compute_metrics --task={self.spec.task} --setup_name={setup} --exp_name={exp} --action={action}")
                            elif any(not (metric_dir / f"{sn}_{sf}").is_file() for sf in metric_suffixes for sn in sns):
                                errors["missing_file"].append(
                                    f"python -m tseval.main exp_compute_metrics --task={self.spec.task} --setup_name={setup} --exp_name={exp} --action={action}")
                    else:
                        warnings["missing_file_possible"].append(f"{setup}, {model}, {exp}")

        for key, values in errors.items():
            print(f"ERROR: {key} -----")
            for value in values:
                print(f"  {value}")
            print("-----")
        for key, values in warnings.items():
            print(f"WARNING: {key} -----")
            for value in values:
                print(f"  {value}")
            print("-----")
        print(f"Execute this dvc command to update results to latest version:")
        print(f"  dvc pull " + " ".join([f"'{str(x)}'" for x in dvc_pull]))

    def recompute_metrics(self):
        commands = []
        dvc_add = []
        for model in self.spec.models:
            for setup in self.spec.setups:
                for exp in model.exps:
                    metric_dir = Macros.work_dir / self.spec.task / "metric" / setup / exp
                    for action in [Macros.val, Macros.test_standard, Macros.test_common]:
                        commands.append(
                            f"python -m tseval.main exp_compute_metrics --task={self.spec.task} --setup_name={setup} --exp_name={exp} --action={action}")
                    dvc_add.append(metric_dir)

        # Prepare a temporary script
        s = "#!/bin/bash\n"
        s += f"source {Environment.get_conda_init_path()}\n"
        s += "conda activate tseval\n"
        s += f"cd {Macros.python_dir}\n"
        s += "set -e\n"
        s += "set -x\n"
        for c in commands:
            s += c + "\n"
        s += "dvc add " + " ".join([str(x) for x in dvc_add])
        script_path = Path(tempfile.mktemp(prefix="tseval"))
        IOUtils.dump(script_path, s, IOUtils.Format.txt)
        script_path.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
        print(script_path)

    @classmethod
    def get_result_prefixes(cls, split: str) -> List[str]:
        return [Macros.val, Macros.test_standard] + \
               [f"{Macros.test_common}-{x}-{y}" for x, y in Macros.get_pairwise_split_types_with(split)]

    def extract_metrics(self):
        lod: List[dict] = []
        for setup in self.spec.setups:
            sns = self.get_result_prefixes(setup)
            for model in self.spec.models:
                for trial_i, exp in enumerate(model.exps):
                    metric_dir = Macros.work_dir / self.spec.task / "metric" / setup / exp
                    for sn in sns:
                        metric_file = metric_dir / f"{sn}_metrics_list.pkl"
                        metrics = IOUtils.load(metric_file, IOUtils.Format.pkl)
                        for m in self.spec.metrics:
                            for data_i, score in enumerate(metrics[m]):
                                lod.append({
                                    "setup": setup,
                                    "model": model.name,
                                    "trial_i": trial_i,
                                    "data_i": data_i,
                                    "set_name": sn,
                                    "metric": m,
                                    "score": score,
                                })

        full_df = pd.DataFrame(lod)
        IOUtils.dump(self.output_dir / "full_df.pkl", full_df, IOUtils.Format.pkl)

        # Compute average
        avg_df = self.get_avg_df(full_df)
        IOUtils.dump(self.output_dir / "avg_df.pkl", avg_df, IOUtils.Format.pkl)

    def extract_data_similarities(self):
        lod: List[dict] = []

        # Load data
        setup2code: Dict[str, Dict[int, List[str]]] = {}
        setup2nl: Dict[str, Dict[int, List[str]]] = {}
        for setup in self.spec.setups:
            setup_dir = Macros.work_dir / self.spec.task / "setup" / setup
            Utils.expect_dir_or_suggest_dvc_pull(setup_dir)

            dataset = MethodData.load_dataset(setup_dir / "data")
            setup2code[setup] = {d.id: ModelUtils.regroup_subtokens(d.code, d.misc["code_src_ids"]) for d in dataset}
            if self.spec.task == Macros.com_gen:
                setup2nl[setup] = {d.id: d.comment_summary for d in dataset}
            elif self.spec.task == Macros.met_nam:
                setup2nl[setup] = {d.id: d.name for d in dataset}
            else:
                raise RuntimeError("Unknown task")

        # Compute similarities for TestS sets
        for setup in self.spec.setups:
            train_code = []
            train_nl = []
            for split in [Macros.train, Macros.val]:
                ids = IOUtils.load(Macros.work_dir / self.spec.task / "setup" / setup / "data" / f"split_{split}.json", IOUtils.Format.json)
                train_code += [setup2code[setup][i] for i in ids]
                train_nl += [setup2nl[setup][i] for i in ids]

            ids = IOUtils.load(Macros.work_dir / self.spec.task / "setup" / setup / "data" / f"split_{Macros.test_standard}.json", IOUtils.Format.json)
            tests_code = [setup2code[setup][i] for i in ids]
            tests_nl = [setup2nl[setup][i] for i in ids]
            for data_i, (code_sim, nl_sim) in enumerate(self.compute_data_similarities(
                    tests_code, tests_nl, train_code, train_nl,
            )):
                lod.append({
                    "setup": setup,
                    "set_name": Macros.test_standard,
                    "data_i": data_i,
                    "code_sim": code_sim,
                    "nl_sim": nl_sim,
                })

        # Compute similarities for TestC sets
        for setup1, setup2 in itertools.combinations(self.spec.setups, 2):
            train_code = []
            train_nl = []
            for setup in [setup1, setup2]:
                for split in [Macros.train, Macros.val]:
                    ids = IOUtils.load(Macros.work_dir / self.spec.task / "setup" / setup / "data" / f"split_{split}.json", IOUtils.Format.json)
                    train_code += [setup2code[setup][i] for i in ids]
                    train_nl += [setup2nl[setup][i] for i in ids]

            ids = IOUtils.load(Macros.work_dir / self.spec.task / "setup" / setup1 / "data" / f"split_{Macros.test_common}-{setup1}-{setup2}.json", IOUtils.Format.json)
            testc_code = [setup2code[setup1][i] for i in ids]
            testc_nl = [setup2nl[setup1][i] for i in ids]
            for data_i, (code_sim, nl_sim) in enumerate(self.compute_data_similarities(
                    testc_code, testc_nl, train_code, train_nl,
            )):
                for setup in [setup1, setup2]:
                    lod.append({
                        "setup": setup,
                        "set_name": f"{Macros.test_common}-{setup1}-{setup2}",
                        "data_i": data_i,
                        "code_sim": code_sim,
                        "nl_sim": nl_sim,
                    })

        # Save the similarities
        data_sim_df = pd.DataFrame(lod)
        IOUtils.dump(self.output_dir / "data_sim_df.pkl", data_sim_df, IOUtils.Format.pkl)

    def filter_near_duplicates_and_analyze(
            self,
            code_sim_threshold: float,
            nl_sim_threshold: float,
            config_name: str,
            only_tables_plots: bool = False,
    ):
        output_dir = self.output_dir / f"nd_{config_name}"

        if not only_tables_plots:
            IOUtils.rm_dir(output_dir)
            output_dir.mkdir(parents=True)

            # Filter data
            data_sim_df = IOUtils.load(self.output_dir / "data_sim_df.pkl", IOUtils.Format.pkl)
            filtered_data_sim_df = data_sim_df.loc[lambda x: (x["code_sim"] < code_sim_threshold) & (x["nl_sim"] < nl_sim_threshold)]
            full_df = IOUtils.load(self.output_dir / "full_df.pkl", IOUtils.Format.pkl)
            filtered_df = full_df.merge(filtered_data_sim_df, on=["setup", "set_name", "data_i"])\
                .drop(["code_sim", "nl_sim"], axis=1)
            IOUtils.dump(output_dir / "filtered_df.pkl", filtered_df, IOUtils.Format.pkl)

            # Compute average
            avg_df = self.get_avg_df(filtered_df)
            IOUtils.dump(output_dir / "avg_df.pkl", avg_df, IOUtils.Format.pkl)

            # Make numbers for data
            table_sub_dir = self.table_dir / f"{self.spec.task}_nd_{config_name}"
            IOUtils.rm_dir(table_sub_dir)
            table_sub_dir.mkdir(parents=True)
            self.make_numbers_nd_dataset(
                table_dir=table_sub_dir,
                spec=self.spec,
                original_data_sim_df=data_sim_df,
                filtered_data_sim_df=filtered_data_sim_df,
                suffix=f"_{config_name}",
            )

            # Significant test
            compare_df = self.sign_test(filtered_df, avg_df, output_dir)
        else:
            table_sub_dir = self.table_dir / f"{self.spec.task}_nd_{config_name}"
            filtered_df = IOUtils.load(output_dir / "filtered_df.pkl", IOUtils.Format.pkl)
            avg_df = IOUtils.load(output_dir / "avg_df.pkl", IOUtils.Format.pkl)
            compare_df = IOUtils.load(output_dir / "compare_df.pkl", IOUtils.Format.pkl)

        # Make tables for results
        self.make_tables(
            table_dir=table_sub_dir,
            spec=self.spec,
            avg_df=avg_df,
            compare_df=compare_df,
            suffix=f"_{config_name}",
        )

        # Make plots
        self.make_plots(
            plot_sub_dir=Macros.paper_dir / "figs" / f"results-{self.spec.task}_nd_{config_name}",
            spec=self.spec,
            full_df=filtered_df,
        )

    @classmethod
    def compute_data_similarities(
            cls,
            eval_code_tokens: List[List[str]],
            eval_nl_tokens: List[List[str]],
            train_code_tokens: List[List[str]],
            train_nl_tokens: List[List[str]],
    ) -> List[Tuple[float, float]]:
        """
        Measures the similarities between each eval set data against the most similar train set data, using the "near_duplicate_similarity" metric.

        The metric computes token level accuracy, and early stops computation if more than 10% tokens are mismatching.
        It is used because of its low computation time; other metrics (e.g., BLEU, edit distance) can take
        prohibited time (48+ hours) on a decent-sized dataset (~70k data).
        """
        similarities: List[Tuple[float, float]] = []

        for i in tqdm(range(len(eval_code_tokens))):
            code_sim = max(EvalMetrics.batch_near_duplicate_similarity(
                [eval_code_tokens[i]] * len(train_code_tokens),
                train_code_tokens,
            ))
            nl_sim = max(EvalMetrics.batch_near_duplicate_similarity(
                [eval_nl_tokens[i]] * len(train_code_tokens),
                train_nl_tokens,
            ))
            similarities.append((code_sim, nl_sim))

        return similarities

    @classmethod
    def get_avg_df(cls, full_df: pd.DataFrame) -> pd.DataFrame:
        return full_df \
            .groupby(["setup", "model", "set_name", "metric"], as_index=False) \
            .mean() \
            .drop(["trial_i", "data_i"], axis=1)

    def sign_test_default(self):
        full_df = IOUtils.load(self.output_dir / "full_df.pkl", IOUtils.Format.pkl)
        avg_df = IOUtils.load(self.output_dir / "avg_df.pkl", IOUtils.Format.pkl)
        self.sign_test(full_df, avg_df, self.output_dir)

    @classmethod
    def sign_test(
            cls,
            full_df: pd.DataFrame,
            avg_df: Optional[pd.DataFrame] = None,
            output_dir: Optional[Path] = None,
    ) -> pd.DataFrame:
        tbar = tqdm("Loading data")
        if avg_df is None:
            avg_df = cls.get_avg_df(full_df)

        compare_lod: List[dict] = []
        tbar.set_description("Performing significance tests")
        total = sum([
            len(list(itertools.combinations(range(len(g)), 2)))
            for g in avg_df.groupby(["set_name", "metric"]).groups.values()
        ])
        tbar.reset(total)
        for (sn, metric), sub_df in full_df.groupby(["set_name", "metric"], as_index=False):
            setup_model_scores: List[Tuple[str, str, List[float]]] = []
            for (setup, model), sub_sub_df in sub_df.groupby(["setup", "model"], as_index=False):
                setup_model_scores.append((
                    setup,
                    model,
                    sub_sub_df.sort_values(["trial_i", "data_i"])["score"].tolist(),
                ))
            for i, j in itertools.combinations(range(len(setup_model_scores)), 2):
                setup_i, model_i, scores_i = setup_model_scores[i]
                setup_j, model_j, scores_j = setup_model_scores[j]
                if setup_i != setup_j and sn not in [
                    f"{Macros.test_common}-{setup_i}-{setup_j}",
                    f"{Macros.test_common}-{setup_j}-{setup_i}",
                ]:
                    tbar.total -= 1
                    continue

                tbar.set_description(f"{sn} {metric}: {setup_i} {model_i} vs. {setup_j} {model_j}")

                i_ge_j = (np.mean(scores_i) >= np.mean(scores_j))
                if i_ge_j:
                    sign_p = bootstrap(scores_i, scores_j)
                else:
                    sign_p = bootstrap(scores_j, scores_i)
                compare_lod.append({
                    "set_name": sn,
                    "metric": metric,
                    "setup_i": setup_i,
                    "model_i": model_i,
                    "setup_j": setup_j,
                    "model_j": model_j,
                    "i_ge_j": i_ge_j,
                    "sign_p": sign_p,
                })
                tbar.update(1)

        compare_df = pd.DataFrame(compare_lod)
        if output_dir is not None:
            IOUtils.dump(output_dir / "compare_df.pkl", compare_df, IOUtils.Format.pkl)
        return compare_df

    def make_tables_default(self):
        avg_df = IOUtils.load(self.output_dir / "avg_df.pkl", IOUtils.Format.pkl)
        compare_df = IOUtils.load(self.output_dir / "compare_df.pkl", IOUtils.Format.pkl)
        self.make_tables(
            table_dir=self.table_dir,
            spec=self.spec,
            avg_df=avg_df,
            compare_df=compare_df,
        )

    @classmethod
    def make_tables(
            cls,
            table_dir: Path,
            spec: ExperimentsSpec,
            avg_df: pd.DataFrame,
            compare_df: pd.DataFrame,
            suffix: str = "",
    ):
        cls.make_numbers_results(table_dir, spec, avg_df, suffix)
        cls.make_table_metrics(table_dir, spec, avg_df, compare_df, suffix)
        cls.make_table_aux_metrics(table_dir, spec, avg_df, compare_df, suffix)

    @classmethod
    def make_numbers_results(
            cls,
            table_dir: Path,
            spec: ExperimentsSpec,
            avg_df: pd.DataFrame,
            suffix: str = "",
    ):
        f = latex.File(table_dir / f"numbers-results-{spec.task}.tex")
        for row in avg_df.itertuples():
            f.append_macro(latex.Macro(
                f"result-{spec.task}_{row.setup}_{row.model}_{row.set_name}_{row.metric}{suffix}",
                f"{row.score * 100:.1f}",
            ))
        f.save()

    @classmethod
    def make_numbers_nd_dataset(
            cls,
            table_dir: Path,
            spec: ExperimentsSpec,
            original_data_sim_df: pd.DataFrame,
            filtered_data_sim_df: pd.DataFrame,
            suffix: str = "",
    ):
        f = latex.File(table_dir / f"numbers-dataset-metrics.tex")

        # The (setup, set_name) pairs to count; avoid double-counting test_common sets
        pairs = [(setup, Macros.test_standard) for setup in spec.setups] + \
            [(setup1, f"{Macros.test_common}-{setup1}-{setup2}") for setup1, setup2 in itertools.combinations(spec.setups, 2)]

        total_num_original, total_num_filtered = 0, 0
        test_standard_num_original, test_standard_num_filtered = 0, 0
        test_common_num_original, test_common_num_filtered = 0, 0
        for (setup, set_name) in pairs:
            original_df = original_data_sim_df[lambda x: (x["setup"] == setup) & (x["set_name"] == set_name)]
            filtered_df = filtered_data_sim_df[lambda x: (x["setup"] == setup) & (x["set_name"] == set_name)]
            num_original = len(original_df)
            total_num_original += num_original
            num_filtered = len(filtered_df)
            total_num_filtered += num_filtered
            if set_name.startswith(Macros.test_standard):
                test_standard_num_original += num_original
                test_standard_num_filtered += num_filtered
            if set_name.startswith(Macros.test_common):
                test_common_num_original += num_original
                test_common_num_filtered += num_filtered
            f.append_macro(latex.Macro(f"ds-{spec.task}-{setup}-{set_name}-original_data{suffix}", f"{num_original:,d}"))
            f.append_macro(latex.Macro(f"ds-{spec.task}-{setup}-{set_name}-filtered_data{suffix}", f"{num_filtered:,d}"))
            f.append_macro(latex.Macro(f"ds-{spec.task}-{setup}-{set_name}-filter{suffix}", f"{num_original - num_filtered:,d}"))
            f.append_macro(latex.Macro(f"ds-{spec.task}-{setup}-{set_name}-filter_pct{suffix}", f"{(num_original - num_filtered) / num_original:.1%}".replace("%", r"\%")))

        for s in ["total", Macros.test_standard, Macros.test_common]:
            num_original = locals()[f"{s}_num_original"]
            num_filtered = locals()[f"{s}_num_filtered"]
            f.append_macro(latex.Macro(f"ds-{spec.task}-{s}-original_data{suffix}", f"{num_original:,d}"))
            f.append_macro(latex.Macro(f"ds-{spec.task}-{s}-filtered_data{suffix}", f"{num_filtered:,d}"))
            f.append_macro(latex.Macro(f"ds-{spec.task}-{s}-filter{suffix}", f"{num_original - num_filtered:,d}"))
            f.append_macro(latex.Macro(f"ds-{spec.task}-{s}-filter_pct{suffix}", f"{(num_original - num_filtered) / num_original:.1%}".replace("%", r"\%")))
            print(f"Filtered {(num_original - num_filtered) / num_original:.1%} of {s} data")

        f.save()

    @classmethod
    def round_scores(cls, df):
        df["score"] = [float(f"{x * 100:.1f}") for x in df["score"]]

    SYMBOLS = [
        r"\alpha", r"\beta", r"\gamma", r"\delta",
        r"\epsilon", r"\zeta", r"\eta", r"\theta",
        r"\iota", r"\kappa", r"\lambda", r"\mu",
        r"\nu", r"\tau", r"\pi", r"\rho",
        r"\sigma", r"\tau", r"\upsilon", r"\phi",
        r"\chi", r"\psi", r"\omega",
    ]

    SIGN_LEVEL = 0.05

    @classmethod
    def make_table_metrics(
            cls,
            table_dir: Path,
            spec: ExperimentsSpec,
            avg_df: pd.DataFrame,
            compare_df: pd.DataFrame,
            suffix: str = "",
    ):
        f = latex.File(table_dir / f"table-results-{spec.task}{suffix}.tex")
        cls.round_scores(avg_df)

        metrics = spec.table_args["metrics"]

        f.append(r"\begin{table}[t]")
        f.append(r"\begin{footnotesize}")
        f.append(r"\begin{center}")
        table_name = f"results-{spec.task}{suffix}"
        f.append(r"\begin{tabular}{@{\hspace{0pt}} l @{\hspace{0pt}} | r@{\hspace{3pt}}r | r@{\hspace{3pt}}r | r@{\hspace{3pt}}r @{\hspace{0pt}}}")
        f.append(r"\toprule")

        f.append(r"\makecell[c]{" + latex.Macro("TH-train-on").use() + "}")
        for s1, s2 in itertools.combinations(Macros.split_types, 2):
            for setup in [s1, s2]:
                f.append(r" & \makecell[c]{" + latex.Macro(f"TH-{setup}").use() + "}")
        f.append(r"\\ \cline{2-3} \cline{4-5} \cline{6-7}")

        f.append(r"\makecell[c]{" + latex.Macro("TH-test-on").use() + "}")
        for s1, s2 in itertools.combinations(Macros.split_types, 2):
            # f.append(r" & \makecell[l]{" + latex.Macro(f"TH-{s1}-{s2}").use() + r"}\hspace{-24pt} & ")
            style = "c|" if s1 != Macros.split_types[-2] else "c"
            f.append(r" & \multicolumn{2}{" + style + "}{" + latex.Macro(f"TH-{s1}-{s2}").use() + "}")
        f.append(r"\\")

        symbol_i = 0
        for metric in metrics:
            symbol_i = cls.add_table_section_metric(f, spec, avg_df, compare_df, metric, symbol_i, suffix)

        f.append(r"\bottomrule")
        f.append(r"\end{tabular}")
        f.append(r"\end{center}")
        f.append(r"\end{footnotesize}")
        f.append(r"\vspace{" + latex.Macro(f"TV-{table_name}").use() + "}")
        f.append(r"\caption{" + latex.Macro(f"TC-{table_name}").use() + r"}")
        f.append(r"\end{table}")

        f.save()

    @classmethod
    def add_table_section_metric(
            cls,
            f: latex.File,
            spec: ExperimentsSpec,
            avg_df: pd.DataFrame,
            compare_df: pd.DataFrame,
            metric: str,
            symbol_i: int,
            suffix: str = "",
    ) -> int:
        f.append(r"\midrule")
        f.append(r"\midrule")
        f.append(r"\multicolumn{7}{c}{" + latex.Macro(f"TH-metric-table-{metric}").use() + r"} \\")
        f.append(r"\midrule")

        # Figure out max values
        key2bold = collections.defaultdict(bool)
        for s1, s2 in itertools.combinations(Macros.split_types, 2):
            for setup in [s1, s2]:
                relevant_df = avg_df.loc[
                    lambda df: (df["metric"] == metric) &
                               (df["set_name"] == f"{Macros.test_common}-{s1}-{s2}") &
                               (df["setup"] == setup)
                ]
                max_score = relevant_df["score"].max()
                for row in relevant_df.loc[lambda df: (df["score"] == max_score)].itertuples():
                    key2bold[(row.setup, row.model, row.set_name)] = True

        # Figure out sign tests prefixes
        key2prefixes = collections.defaultdict(list)
        for s1, s2 in itertools.combinations(Macros.split_types, 2):
            for row in compare_df.loc[
                lambda df: (df["metric"] == metric) & (df["set_name"] == f"{Macros.test_common}-{s1}-{s2}")
            ].itertuples():
                if row.sign_p > cls.SIGN_LEVEL:
                    symbol = cls.SYMBOLS[symbol_i]
                    symbol_i += 1
                    key2prefixes[(row.setup_i, row.model_i, row.set_name)].insert(0, symbol)
                    key2prefixes[(row.setup_j, row.model_j, row.set_name)].insert(0, symbol)

        for model in spec.models:
            f.append(latex.Macro(f"TH-model-{model.name}").use())
            for s1, s2 in itertools.combinations(Macros.split_types, 2):
                for setup in [s1, s2]:
                    number = latex.Macro(
                        f"result-{spec.task}_{setup}_{model.name}_{Macros.test_common}-{s1}-{s2}_{metric}{suffix}").use()
                    # Add modifiers (bold, prefixes)
                    if key2bold[(setup, model.name, f"{Macros.test_common}-{s1}-{s2}")]:
                        number = r"\textbf{" + number + "}"
                    for prefix in key2prefixes[(setup, model.name, f"{Macros.test_common}-{s1}-{s2}")]:
                        number = r"$^{" + prefix + r"}$" + number

                    f.append(r" & " + number)

            f.append(r"\\")
        return symbol_i

    @classmethod
    def make_table_aux_metrics(
            cls,
            table_dir: Path,
            spec: ExperimentsSpec,
            avg_df: pd.DataFrame,
            compare_df: pd.DataFrame,
            suffix: str = "",
    ):
        f = latex.File(table_dir / f"table-results-aux-{spec.task}{suffix}.tex")
        cls.round_scores(avg_df)

        metrics = spec.table_args["metrics"]

        f.append(r"\begin{table}[t]")
        f.append(r"\begin{footnotesize}")
        f.append(r"\begin{center}")
        table_name = f"results-aux-{spec.task}{suffix}"
        f.append(r"\begin{tabular}{@{\hspace{0pt}} l @{\hspace{0pt}} | @{\hspace{2pt}}r@{\hspace{2pt}} | @{\hspace{2pt}}r@{\hspace{2pt}} | r@{\hspace{2pt}} | @{\hspace{2pt}}r@{\hspace{2pt}} | r@{\hspace{2pt}} | @{\hspace{2pt}}r @{\hspace{0pt}}}")
        f.append(r"\toprule")

        f.append(r"\makecell[c]{" + latex.Macro("TH-train-on").use() + "}")
        for s in Macros.split_types:
            style = "c|" if s != Macros.split_types[-1] else "c"
            f.append(r" & \multicolumn{2}{" + style + "}{" + latex.Macro(f"TH-{s}").use() + "}")
        f.append(r"\\ \cline{2-3} \cline{4-5} \cline{6-7}")

        f.append(r"\makecell[c]{" + latex.Macro("TH-test-on").use() + "}")
        for s in Macros.split_types:
            f.append(r" & " + latex.Macro(f"TH-{Macros.val}").use())
            f.append(r" & " + latex.Macro(f"TH-{Macros.test_standard}").use())
        f.append(r"\\")

        symbol_i = 0
        for metric in metrics:
            symbol_i = cls.add_table_aux_section_metric(f, spec, avg_df, compare_df, metric, symbol_i, suffix)

        f.append(r"\bottomrule")
        f.append(r"\end{tabular}")
        f.append(r"\end{center}")
        f.append(r"\end{footnotesize}")
        f.append(r"\vspace{" + latex.Macro(f"TV-{table_name}").use() + "}")
        f.append(r"\caption{" + latex.Macro(f"TC-{table_name}").use() + r"}")
        f.append(r"\end{table}")

        f.save()

    @classmethod
    def add_table_aux_section_metric(
            cls,
            f: latex.File,
            spec: ExperimentsSpec,
            avg_df: pd.DataFrame,
            compare_df: pd.DataFrame,
            metric: str,
            symbol_i: int,
            suffix: str = "",
    ):
        f.append(r"\midrule")
        f.append(r"\midrule")
        f.append(r"\multicolumn{7}{c}{" + latex.Macro(f"TH-metric-table-{metric}").use() + r"} \\")
        f.append(r"\midrule")

        # Figure out max values
        key2bold = collections.defaultdict(bool)
        for setup in Macros.split_types:
            for sn in [Macros.val, Macros.test_standard]:
                relevant_df = avg_df.loc[
                    lambda df: (df["metric"] == metric) & (df["set_name"] == sn) & (df["setup"] == setup)]
                max_score = relevant_df["score"].max()
                for row in relevant_df.loc[lambda df: (df["score"] == max_score)].itertuples():
                    key2bold[(row.setup, row.model, row.set_name)] = True

        # Figure out sign tests prefixes
        key2prefixes = collections.defaultdict(list)
        for sn in [Macros.val, Macros.test_standard]:
            for row in compare_df.loc[lambda df: (df["metric"] == metric) & (df["set_name"] == sn)].itertuples():
                if row.sign_p > cls.SIGN_LEVEL:
                    symbol = cls.SYMBOLS[symbol_i]
                    symbol_i += 1
                    key2prefixes[(row.setup_i, row.model_i, row.set_name)].insert(0, symbol)
                    key2prefixes[(row.setup_j, row.model_j, row.set_name)].insert(0, symbol)

        for model in spec.models:
            f.append(latex.Macro(f"TH-model-{model.name}").use())
            for setup in Macros.split_types:
                for sn in [Macros.val, Macros.test_standard]:
                    number = latex.Macro(f"result-{spec.task}_{setup}_{model.name}_{sn}_{metric}{suffix}").use()
                    # Add modifiers (bold, prefixes)
                    if key2bold[(setup, model.name, sn)]:
                        number = r"\textbf{" + number + "}"
                    for prefix in key2prefixes[(setup, model.name, sn)]:
                        number = r"$^{" + prefix + r"}$" + number

                    f.append(r" & " + number)

            f.append(r"\\")
        return symbol_i

    def make_plots_default(self):
        full_df = IOUtils.load(self.output_dir / "full_df.pkl", IOUtils.Format.pkl)
        self.make_plots(
            plot_sub_dir=self.plot_sub_dir,
            spec=self.spec,
            full_df=full_df,
        )

    @classmethod
    def make_plots(
            cls,
            plot_sub_dir: Path,
            spec: ExperimentsSpec,
            full_df: pd.DataFrame,
    ):
        from tseval.Plot import Plot
        IOUtils.rm_dir(plot_sub_dir)
        plot_sub_dir.mkdir(parents=True, exist_ok=True)
        Plot.init_plot_libs()
        cls.make_plot_metrics_bars(plot_sub_dir, spec, full_df)

    @classmethod
    def make_plot_metrics_bars(
            cls,
            plot_sub_dir: Path,
            spec: ExperimentsSpec,
            full_df: pd.DataFrame,
    ):
        metrics_order = spec.plot_args["metrics"].keys()
        hue_order = [model.name for model in spec.models]

        # Process the metrics that should be shown in percentage
        for metric in metrics_order:
            if spec.plot_args["metrics_percent"].get(metric, False):
                full_df.loc[lambda x: x["metric"] == metric, "score"] = \
                    full_df.loc[lambda x: x["metric"] == metric, "score"] * 100

        # Draw plots
        extracted_legend = False
        for (setup, set_name, metric), df in full_df.groupby(["setup", "set_name", "metric"], as_index=False):
            if metric not in metrics_order:
                continue
            fig: plt.Figure = plt.figure(figsize=(5, 3))
            ax: plt.Axes = fig.add_subplot()
            sns.barplot(
                ax=ax,
                data=df,
                orient="h",
                x="score",
                y="metric",
                hue="model", hue_order=hue_order,
                ci=None,
            )
            ax.set_xlabel(None)
            ax.set_ylabel(None)
            ax.set_yticklabels([""])
            if spec.plot_args["metrics_percent"].get(metric, False):
                ax.set_xlim(0, 100)
            else:
                ax.set_xlim(0, 1)

            # Extract legend
            if not extracted_legend:
                fig_legend: plt.Figure = plt.figure()
                legend = fig_legend.legend(handles=ax.legend_.legendHandles,
                    labels=[spec.plot_args["models"][model.name] for model in spec.models],
                    ncol=len(spec.models))
                fig_legend.savefig(
                    plot_sub_dir / f"legend.pdf",
                    bbox_extra_artists=(legend,),
                    bbox_inches="tight",
                )
                extracted_legend = True
                plt.close(fig_legend)

            ax.get_legend().remove()
            fig.tight_layout()
            fig.savefig(plot_sub_dir / f"bar-{setup}-{set_name}-{metric}.pdf")
            plt.close(fig)

        # Generate tex files that organizes the plots
        cls.gen_plot_tex_file(plot_sub_dir, metrics_order)
        cls.gen_plot_aux_tex_file(plot_sub_dir, metrics_order)

    @classmethod
    def gen_plot_tex_file(cls, plot_sub_dir: Path, metrics_order: List[str]):
        plot_sub_dir_rel = str(plot_sub_dir.relative_to(Macros.paper_dir))

        f = latex.File(plot_sub_dir / f"plot.tex")
        f.append(r"\begin{center}")
        f.append(r"\begin{footnotesize}")
        f.append(r"\begin{tabular}{|r|@{}c@{}c@{}|@{}c@{}c@{}|@{}c@{}c@{}|}")
        f.append(r"\hline")

        f.append(r"\makecell[c]{" + latex.Macro("TH-train-on").use() + "}")
        for s1, s2 in itertools.combinations(Macros.split_types, 2):
            for setup in [s1, s2]:
                f.append(r" & \makecell[c]{" + latex.Macro(f"TH-{setup}").use() + "}")
        f.append(r"\\")
        f.append(r"\hline")

        f.append(r"\makecell[c]{" + latex.Macro("TH-test-on").use() + "}")
        for s1, s2 in itertools.combinations(Macros.split_types, 2):
            style = "c|"
            f.append(r" & \multicolumn{2}{" + style + "}{" + latex.Macro(f"TH-{s1}-{s2}").use() + "}")
        f.append(r"\\")

        f.append(r"\hline")

        for metric in metrics_order:
            f.append(latex.Macro(f"TH-metric-{metric}").use())
            for s1, s2 in itertools.combinations(Macros.split_types, 2):
                for setup in [s1, s2]:
                    f.append(r" & \begin{minipage}{.12\textwidth}\includegraphics[width=\textwidth]{"
                             + f"{plot_sub_dir_rel}/bar-{setup}-{Macros.test_common}-{s1}-{s2}-{metric}"
                             + r"}\end{minipage}")
            f.append(r"\\")
        f.append(r"\hline")
        f.append(r"\multicolumn{7}{c}{\includegraphics[scale=0.3]{" + f"{plot_sub_dir_rel}/legend" + r"}} \\")
        f.append(r"\end{tabular}")
        f.append(r"\end{footnotesize}")
        f.append(r"\end{center}")
        f.save()

    @classmethod
    def gen_plot_aux_tex_file(cls, plot_sub_dir: Path, metrics_order: List[str]):
        plot_sub_dir_rel = str(plot_sub_dir.relative_to(Macros.paper_dir))
        f = latex.File(plot_sub_dir / f"plot-aux.tex")
        f.append(r"\begin{center}")
        f.append(r"\begin{footnotesize}")
        f.append(r"\begin{tabular}{|r|c|c|c|c|c|c|}")
        f.append(r"\hline")

        f.append(r"\makecell[c]{" + latex.Macro("TH-train-on").use() + "}")
        for s in Macros.split_types:
            style = "c|"
            f.append(r" & \multicolumn{2}{" + style + "}{" + latex.Macro(f"TH-{s}").use() + "}")
        f.append(r"\\")
        f.append(r"\hline")

        f.append(r"\makecell[c]{" + latex.Macro("TH-test-on").use() + "}")
        for s in Macros.split_types:
            f.append(r" & " + latex.Macro(f"TH-{Macros.val}").use())
            f.append(r" & " + latex.Macro(f"TH-{Macros.test_standard}").use())
        f.append(r"\\")

        f.append(r"\hline")

        for metric in metrics_order:
            f.append(latex.Macro(f"TH-metric-{metric}").use())
            for setup in Macros.split_types:
                for sn in [Macros.val, Macros.test_standard]:
                    f.append(r" & \begin{minipage}{.12\textwidth}\includegraphics[width=\textwidth]{"
                             + f"{plot_sub_dir_rel}/bar-{setup}-{sn}-{metric}"
                             + r"}\end{minipage}")
            f.append(r"\\")
        f.append(r"\hline")
        f.append(r"\multicolumn{7}{c}{\includegraphics[scale=0.3]{" + f"{plot_sub_dir_rel}/legend" + r"}} \\")
        f.append(r"\end{tabular}")
        f.append(r"\end{footnotesize}")
        f.append(r"\end{center}")
        f.save()

    def sample_results(self, seed: int = 7, count: int = 100):
        tbar = tqdm("Sample results",
            total=len(self.spec.setups) + len(list(itertools.combinations(self.spec.setups, 2))))
        for setup in self.spec.setups:
            self.sample_results_one(Macros.val, [setup], seed, count)
            tbar.update(1)

        for s1, s2 in itertools.combinations(self.spec.setups, 2):
            self.sample_results_one(f"{Macros.test_common}-{s1}-{s2}", [s1, s2], seed, count)
            tbar.update(1)

    def sample_results_one(self, set_name: str, setups: List[str], seed: int = 7, count: int = 100):
        eval_ids = IOUtils.load(
            Macros.work_dir / self.spec.task / "setup" / setups[0] / "data" / f"split_{set_name}.json",
            IOUtils.Format.json)
        dataset: List[MethodData] = MethodData.load_dataset(
            Macros.work_dir / self.spec.task / "setup" / setups[0] / "data", expected_ids=eval_ids)

        # Ensure dataset is in the order of eval_ids
        indexed_dataset = {d.id: d for d in dataset}
        dataset = [indexed_dataset[i] for i in eval_ids]

        # Collect model results & metrics
        key2results: Dict[Tuple[str, str], List[List[str]]] = {}
        key2metrics: Dict[Tuple[str, str], Dict[str, List[List[float]]]] = {}
        for setup in setups:
            for model in self.spec.models:
                for exp in model.exps:
                    key2results[(setup, exp)] = IOUtils.load(
                        Macros.work_dir / self.spec.task / "result" / setup / exp / f"{set_name}_predictions.jsonl",
                        IOUtils.Format.jsonList)
                    metrics_list = IOUtils.load(
                        Macros.work_dir / self.spec.task / "metric" / setup / exp / f"{set_name}_metrics_list.pkl",
                        IOUtils.Format.pkl)
                    key2metrics[(setup, exp)] = {}
                    for metric in self.spec.metrics:
                        key2metrics[(setup, exp)][metric] = metrics_list[metric]

        # Sample
        random.seed(seed)
        indices = list(range(len(dataset)))
        random.shuffle(indices)

        if len(setups) == 1:
            sample_file = self.output_dir / "sample" / f"{set_name}-{setups[0]}.java"
        else:
            sample_file = self.output_dir / "sample" / f"{set_name}.java"
        sample_file.parent.mkdir(parents=True, exist_ok=True)
        with open(sample_file, "w") as f:
            f.write(f"// Sample {count}/{len(indices)} data, seed {seed}\n")
            for count_i, i in enumerate(indices[:count]):
                data = dataset[i]
                f.write(f"// ----- {count_i + 1}/{count} | Data #{data.id} -----\n")
                if self.spec.task == "CG":
                    f.write(f"/** {data.misc['orig_comment_summary']} */\n")
                    f.write(data.misc["orig_code"] + "\n")
                    f.write(f"/** {' '.join(data.comment_summary)} */\n")
                else:
                    f.write(f"/** {data.misc['orig_name']} */\n")
                    f.write(data.misc["orig_code_masked"] + "\n")
                    f.write(f"/** {' '.join(data.name)} */\n")
                for setup, exp in key2results.keys():
                    f.write(f"// vvv {setup} {exp}\n")
                    f.write(f"/** {' '.join(key2results[(setup, exp)][i])} */\n")
                    f.write(f"//     ")
                    for metric in self.spec.metrics:
                        f.write(f"{metric}: {key2metrics[(setup, exp)][metric][i] * 100:5.1f} ")
                    f.write("\n")
                f.write("\n\n")
                f.flush()
