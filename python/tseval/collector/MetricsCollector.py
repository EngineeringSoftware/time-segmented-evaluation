import collections
import itertools
import re
import traceback
from typing import *

import javalang
import numpy as np
from nltk.tokenize import word_tokenize
from seutil import IOUtils, LoggingUtils
from tqdm import tqdm

from tseval.data.MethodData import MethodData
from tseval.Environment import Environment
from tseval.Macros import Macros
from tseval.Utils import Utils

logger = LoggingUtils.get_logger(__name__, LoggingUtils.DEBUG if Environment.is_debug else LoggingUtils.INFO)


class MetricsCollector:

    def __init__(self):
        self.output_dir = Macros.results_dir / "metrics"
        IOUtils.mk_dir(self.output_dir)
        return

    def collect_metrics(self, **options):
        which = options["which"]

        if which == "split-dataset":
            self.split_dataset(
                split=options["split"],
            )
        elif which == "setup-dataset":
            self.setup_dataset(
                task=options["task"],
                setup=options["setup"],
            )
        elif which == "setup-dataset-leak":
            self.setup_dataset_leak(
                task=options["task"],
                setup=options["setup"],
            )
        elif which == "raw-data-filtered":
            self.raw_data_filtered()
        else:
            logger.warning(f"No such metrics {which}")

    def split_dataset(self, split: str):
        shared_data_dir = Macros.work_dir / "shared"
        split_dir = Macros.work_dir / "split" / split

        dataset: List[MethodData] = MethodData.load_dataset(shared_data_dir)
        indexed_dataset: Dict[int, MethodData] = {d.id: d for d in dataset}

        metrics = IOUtils.load(split_dir / "stats.json", IOUtils.Format.json)
        metrics["num_prj_all"] = sum(metrics[f"num_prj_{sn}"] for sn in [Macros.train, Macros.val, Macros.test])
        metrics_list: Dict[str, List] = {}

        tbar = tqdm(desc="Computing metrics...", total=1+3*3+6)

        # full set
        x_metrics, x_metrics_list = self.collect_metrics_dataset(dataset)
        for k, v in x_metrics.items():
            metrics[f"all_{k}"] = v
        for k, l in x_metrics_list.items():
            metrics_list[f"all_{k}"] = l
        tbar.update(1)

        # splits
        for s2 in [Macros.train, Macros.val, Macros.test_standard, Macros.test_common]:
            if s2 == Macros.test_common:
                s1_options = [f"{x}-{y}" for x, y in itertools.combinations(Macros.split_types, 2)]
            else:
                s1_options = Macros.split_types
            for s1 in s1_options:
                ids = IOUtils.load(split_dir / f"{s1}-{s2}.json", IOUtils.Format.json)
                x_metrics, x_metrics_list = self.collect_metrics_dataset(
                    [indexed_dataset[i] for i in ids]
                )
                for k, v in x_metrics.items():
                    metrics[f"{s1}-{s2}_{k}"] = v
                for k, l in x_metrics_list.items():
                    metrics_list[f"{split}-{s2}_{k}"] = l
                tbar.update(1)
        tbar.close()

        # Save results
        IOUtils.dump(self.output_dir / f"split-dataset-metrics_{split}.json", metrics, IOUtils.Format.jsonNoSort)
        IOUtils.dump(self.output_dir / f"split-dataset-metrics-list_{split}.pkl", metrics_list, IOUtils.Format.pkl)
        print(Utils.suggest_dvc_add(self.output_dir / f"split-dataset-metrics-list_{split}.pkl"))

    def setup_dataset(self, task: str, setup: str):
        data_dir = Macros.work_dir / task / "setup" / setup / "data"
        dataset: List[MethodData] = MethodData.load_dataset(data_dir)
        indexed_dataset: Dict[int, MethodData] = {d.id: d for d in dataset}

        split2ids = {}
        for split_file in data_dir.glob("split_*.json"):
            split_name = split_file.stem[len("split_"):]
            split2ids[split_name] = IOUtils.load(split_file, IOUtils.Format.json)

        metrics = {}
        metrics_list = {}

        tbar = tqdm(desc="Computing metrics...", total=1+len(split2ids))

        if task == "CG":
            only = ["code", "comment"]
        elif task == "MN":
            only = ["code", "name"]
        else:
            raise RuntimeError()

        # Full set
        x_metrics, x_metrics_list = self.collect_metrics_dataset(dataset, subtokenized=True, only=only)
        for k, v in x_metrics.items():
            metrics[f"all_{k}"] = v
        for k, l in x_metrics_list.items():
            metrics_list[f"all_{k}"] = l
        tbar.update(1)

        # Splits
        for split, ids in split2ids.items():
            x_metrics, x_metrics_list = self.collect_metrics_dataset(
                [indexed_dataset[i] for i in ids],
                subtokenized=True,
                only=only,
            )
            for k, v in x_metrics.items():
                metrics[f"{split}_{k}"] = v
            for k, l in x_metrics_list.items():
                metrics_list[f"{split}_{k}"] = l
            tbar.update(1)
        tbar.close()

        # Save results
        IOUtils.dump(self.output_dir / f"setup-dataset-metrics_{task}_{setup}.json", metrics, IOUtils.Format.jsonNoSort)
        IOUtils.dump(self.output_dir / f"setup-dataset-metrics-list_{task}_{setup}.pkl", metrics_list, IOUtils.Format.pkl)
        print(Utils.suggest_dvc_add(self.output_dir / f"setup-dataset-metrics-list_{task}_{setup}.pkl"))

    @classmethod
    def length_code(cls, code: str) -> int:
        try:
            return len(list(javalang.tokenizer.tokenize(code)))
        except KeyboardInterrupt:
            raise
        except:
            traceback.print_exc()
            return np.NaN

    @classmethod
    def length_nl(cls, nl: str) -> int:
        try:
            return len(list(word_tokenize(nl)))
        except KeyboardInterrupt:
            raise
        except:
            traceback.print_exc()
            return np.NaN

    @classmethod
    def subtokenize_code(cls, tokens: List[str]) -> List[str]:
        """Subtokenize the code."""
        subtokens = list()
        for token in tokens:
            curr = re.sub('([a-z0-9])([A-Z])', r'\1 \2', token).split()
            subtokens = subtokens + [c.lower() for c in curr]
        return subtokens

    @classmethod
    def length_name(cls, name: str) -> int:
        try:
            return len(cls.subtokenize_code([name]))
        except KeyboardInterrupt:
            raise
        except:
            traceback.print_exc()
            return np.NaN

    @classmethod
    def collect_metrics_dataset(
            cls,
            dataset: List[MethodData],
            subtokenized: bool = False,
            only: Optional[List[str]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, List]]:
        metrics_list: Dict[str, List] = {}
        metrics = {}
        metrics[f"num-data"] = len(dataset)

        if only is None or "code" in only:
            if subtokenized:
                metrics_list["len-code"] = [len(d.code) for d in dataset]
            else:
                metrics_list["len-code"] = [cls.length_code(d.code) for d in dataset]

            for k in [100, 150, 200]:
                metrics[f"len-code-le-{k}"] = 100 * (len([n for n in metrics_list["len-code"] if n <= k]) / len(metrics_list["len-code"]))

        if only is None or "comment" in only:
            if subtokenized:
                metrics_list["len-comment"] = [len(d.comment_summary) for d in dataset]
            else:
                metrics_list["len-comment"] = [cls.length_nl(d.comment_summary) for d in dataset]

            for k in [20, 30, 50]:
                metrics[f"len-comment-le-{k}"] = 100 * (len([n for n in metrics_list["len-comment"] if n <= k]) / len(metrics_list["len-comment"]))

        if only is None or "name" in only:
            if subtokenized:
                metrics_list["len-name"] = [len(d.name) for d in dataset]
            else:
                metrics_list["len-name"] = [cls.length_name(d.name) for d in dataset]

            for k in [2, 3, 6]:
                metrics[f"len-name-le-{k}"] = 100 * (len([n for n in metrics_list["len-name"] if n <= k]) / len(metrics_list["len-name"]))

        for k, l in metrics_list.items():
            for s, func in Utils.SUMMARIES_FUNCS.items():
                metrics[f"{k}-{s}"] = func(l)

        return metrics, metrics_list

    def setup_dataset_leak(self, task: str, setup: str):
        setup_dir = Macros.work_dir / task / "setup" / setup
        data_dir = setup_dir / "data"

        metrics = {}

        tbar = tqdm("Loading data")
        dataset: List[MethodData] = MethodData.load_dataset(data_dir, tbar=tbar)
        tbar.close()
        indexed_ds: Dict[int, MethodData] = {d.id: d for d in dataset}

        eval_sns = [Macros.val, Macros.test_standard]
        # Assuming setup's name is the split's name
        for s1, s2 in Macros.get_pairwise_split_types_with(setup):
            eval_sns.append(f"{Macros.test_common}-{s1}-{s2}")
        all_sns = [Macros.train] + eval_sns

        sn2ds = {}
        for ns in all_sns:
            ns_ids = IOUtils.load(data_dir / f"split_{ns}.json")
            sn2ds[ns] = [indexed_ds[i] for i in ns_ids]
            metrics[f"{ns}_num_data"] = len(sn2ds[ns])

        counters: Dict[str, int] = {}
        for a in all_sns:
            for b in ["dup_src", "dup_tgt", "dup_pair"]:
                counters[f"{a}_{b}"] = 0
            if a != Macros.train:
                for b in ["leak_src", "leak_tgt", "leak_pair"]:
                    counters[f"{a}_{b}"] = 0
        sn2pairs: Dict[str, Set[Tuple[str, str]]] = collections.defaultdict(set)
        sn2srcs: Dict[str, Set[str]] = collections.defaultdict(set)
        sn2tgts: Dict[str, Set[str]] = collections.defaultdict(set)

        for d in sn2ds[Macros.train]:
            src = " ".join(d.code)
            if task == Macros.com_gen:
                tgt = " ".join(d.comment_summary)
            elif task == Macros.met_nam:
                tgt = " ".join(d.name)
            else:
                raise RuntimeError()

            if src in sn2srcs[Macros.train]:
                counters[f"{Macros.train}_dup_src"] += 1
            else:
                sn2srcs[Macros.train].add(src)
            if tgt in sn2tgts[Macros.train]:
                counters[f"{Macros.train}_dup_tgt"] += 1
            else:
                sn2tgts[Macros.train].add(tgt)
            if (src, tgt) in sn2pairs[Macros.train]:
                counters[f"{Macros.train}_dup_pair"] += 1
            else:
                sn2pairs[Macros.train].add((src, tgt))

        for sn in eval_sns:
            for d in sn2ds[sn]:
                src = " ".join(d.code)
                if task == Macros.com_gen:
                    tgt = " ".join(d.comment_summary)
                elif task == Macros.met_nam:
                    tgt = " ".join(d.name)
                else:
                    raise RuntimeError()

                if src in sn2srcs[Macros.train]:
                    counters[f"{sn}_leak_src"] += 1
                if src in sn2srcs[sn]:
                    counters[f"{sn}_dup_src"] += 1
                else:
                    sn2srcs[sn].add(src)
                if tgt in sn2tgts[Macros.train]:
                    counters[f"{sn}_leak_tgt"] += 1
                if tgt in sn2tgts[sn]:
                    counters[f"{sn}_dup_tgt"] += 1
                else:
                    sn2tgts[sn].add(tgt)
                if (src, tgt) in sn2pairs[Macros.train]:
                    counters[f"{sn}_leak_pair"] += 1
                if (src, tgt) in sn2pairs[sn]:
                    counters[f"{sn}_dup_pair"] += 1
                else:
                    sn2pairs[sn].add((src, tgt))

        # Compute fractions
        metrics.update(counters)
        for sn in all_sns:
            for k, v in counters.items():
                if k.startswith(sn):
                    metrics[f"{k}_frac"] = counters[k] / metrics[f"{sn}_num_data"]

        # Save results
        IOUtils.dump(self.output_dir / f"setup-dataset-leak-metrics_{task}_{setup}.json", metrics, IOUtils.Format.jsonNoSort)

    def raw_data_filtered(self):
        filtered_counters = collections.Counter()

        for d in Macros.raw_data_dir.glob("*"):
            if not d.is_dir() or not (d / "filtered-counters.json").is_file():
                continue
            prj_filtered_counters = IOUtils.load(d / "filtered-counters.json", IOUtils.Format.json)
            filtered_counters.update(prj_filtered_counters)

        IOUtils.dump(self.output_dir / "raw-data-filtered.json", filtered_counters, IOUtils.Format.jsonPretty)
