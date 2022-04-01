import collections
import copy
import time
from pathlib import Path
from typing import Dict, List

import nltk
import numpy as np
from seutil import IOUtils, LoggingUtils
from tqdm import tqdm

from tseval.comgen.eval.CGModelLoader import CGModelLoader
from tseval.comgen.model.CGModelBase import CGModelBase
from tseval.data.MethodData import MethodData
from tseval.eval.EvalMetrics import EvalMetrics
from tseval.eval.EvalSetupBase import EvalSetupBase
from tseval.Macros import Macros
from tseval.util.ModelUtils import ModelUtils
from tseval.util.TrainConfig import TrainConfig
from tseval.Utils import Utils

logger = LoggingUtils.get_logger(__name__)


class StandardSetup(EvalSetupBase):

    # Validation set on self's split type
    EVAL_VAL = Macros.val
    # Test_standard set on self's split type
    EVAL_TESTS = Macros.test_standard
    # Test_common sets, pairwisely between self's split type and other split types
    EVAL_TESTC = Macros.test_common

    EVAL_ACTIONS = [EVAL_VAL, EVAL_TESTS, EVAL_TESTC]
    DEFAULT_EVAL_ACTION = EVAL_TESTC

    def __init__(
            self,
            work_dir: Path,
            work_subdir: Path,
            setup_name: str,
            split_name: str,
            split_type: str,
    ):
        super().__init__(work_dir, work_subdir, setup_name)
        self.split_name = split_name
        self.split_type = split_type

    def prepare(self) -> None:
        # Check and prepare directories
        split_dir = self.get_split_dir(self.split_name)
        Utils.expect_dir_or_suggest_dvc_pull(self.shared_data_dir)
        Utils.expect_dir_or_suggest_dvc_pull(split_dir)
        IOUtils.rm_dir(self.data_dir)
        self.data_dir.mkdir(parents=True)

        # Copy split indexes
        all_indexes = []
        for sn in [Macros.train, Macros.val, Macros.test_standard]:
            ids = IOUtils.load(split_dir / f"{self.split_type}-{sn}.json", IOUtils.Format.json)
            all_indexes += ids
            IOUtils.dump(self.data_dir / f"split_{sn}.json", ids, IOUtils.Format.json)
        for s1, s2 in Macros.get_pairwise_split_types_with(self.split_type):
            ids = IOUtils.load(split_dir / f"{s1}-{s2}-{Macros.test_common}.json", IOUtils.Format.json)
            all_indexes += ids
            IOUtils.dump(
                self.data_dir / f"split_{Macros.test_common}-{s1}-{s2}.json",
                ids,
                IOUtils.Format.json,
            )
        all_indexes = list(sorted(set(all_indexes)))

        # Load raw data
        tbar = tqdm()
        dataset: List[MethodData] = MethodData.load_dataset(
            self.shared_data_dir,
            expected_ids=all_indexes,
            only=["code", "comment_summary", "misc"],
            tbar=tbar,
        )
        tbar.close()

        # Subtokenize code and comments
        tbar = tqdm()

        tbar.set_description("Subtokenizing")
        tbar.reset(len(dataset))
        orig_code_list = []
        tokenized_comment_list = []
        for d in dataset:
            d.fill_none()
            d.misc["orig_code"] = d.code
            d.misc["orig_comment_summary"] = d.comment_summary
            orig_code_list.append(d.code)
            tokenized_comment_list.append(nltk.word_tokenize(d.comment_summary))
            tbar.update(1)

        tokenized_code_list = ModelUtils.tokenize_javaparser_batch(orig_code_list, dup_share=False, tbar=tbar)

        tbar.set_description("Subtokenizing")
        tbar.reset(len(dataset))
        for d, tokenized_code, tokenized_comment in zip(dataset, tokenized_code_list, tokenized_comment_list):
            d.code, d.misc["code_src_ids"] = ModelUtils.subtokenize_batch(tokenized_code)
            d.comment_summary, d.misc["comment_summary_src_ids"] = ModelUtils.subtokenize_batch(tokenized_comment)

            # convert comment to lower case
            d.comment_summary = [t.lower() for t in d.comment_summary]
            tbar.update(1)
        tbar.close()

        # Clean eval ids
        indexed_dataset = {d.id: d for d in dataset}
        for sn in [Macros.val, Macros.test_standard] + [f"{Macros.test_common}-{x}-{y}" for x, y in Macros.get_pairwise_split_types_with(self.split_type)]:
            eval_ids = IOUtils.load(self.data_dir / f"split_{sn}.json", IOUtils.Format.json)
            IOUtils.dump(self.data_dir / f"split_{sn}.json", self.clean_eval_set(indexed_dataset, eval_ids), IOUtils.Format.json)

        # Save dataset
        MethodData.save_dataset(dataset, self.data_dir)

    def clean_eval_set(self, indexed_dataset: Dict[int, MethodData], eval_ids: List[int]) -> List[int]:
        """
        Keeps the eval set absolutely clean by:
          - Remove duplicate (comment, code) pairs;
          - Remove pseudo-empty comment;
        indexed_dataset should already been subtokenized.
        """
        seen_data = set()
        clean_eval_ids = []
        for i in eval_ids:
            data = indexed_dataset[i]

            # Remove comment like "."
            if len(data.comment_summary) == 1 and data.comment_summary[0] == ".":
                continue

            # Remove duplicate (comment, code) pairs
            data_key = (tuple(data.code), tuple(data.comment_summary))
            if data_key in seen_data:
                continue
            else:
                seen_data.add(data_key)

            clean_eval_ids.append(i)
        return clean_eval_ids

    def train(self, exp_name: str, model_name: str, cont_train: bool, no_save: bool, **options) -> None:
        # Init or load model
        exp_dir = self.get_exp_dir(exp_name)
        train_config = TrainConfig.get_train_config_from_cmd_options(options)
        model = CGModelLoader.init_or_load_model(model_name, exp_dir, cont_train, no_save, options)
        if not no_save:
            IOUtils.dump(exp_dir / "train_config.jsonl", [IOUtils.jsonfy(train_config)], IOUtils.Format.jsonList, append=True)

        # Load data
        tbar = tqdm(desc="Loading data")
        dataset = MethodData.load_dataset(self.data_dir, tbar=tbar)
        indexed_dataset = {d.id: d for d in dataset}

        tbar.set_description("Loading data | take indexes")
        tbar.reset(2)

        train_ids = IOUtils.load(self.data_dir / f"split_{Macros.train}.json", IOUtils.Format.json)
        train_dataset = [indexed_dataset[i] for i in train_ids]
        tbar.update(1)

        val_ids = IOUtils.load(self.data_dir / f"split_{Macros.val}.json", IOUtils.Format.json)
        val_dataset = [indexed_dataset[i] for i in val_ids]
        tbar.update(1)

        tbar.close()

        # Train model
        start = time.time()
        model.train(train_dataset, val_dataset, resources_path=self.data_dir, train_config=train_config)
        end = time.time()

        if not no_save:
            model.save()
            IOUtils.dump(exp_dir / "train_time.json", end - start, IOUtils.Format.json)

    def eval_one(self, exp_name: str, eval_ids: List[int], prefix: str, indexed_dataset: Dict[int, MethodData], model: CGModelBase, gpu_id: int = 0):
        # Prepare output directory
        result_dir = self.get_result_dir(exp_name)
        result_dir.mkdir(parents=True, exist_ok=True)

        # Prepare eval data (remove target)
        eval_dataset = [indexed_dataset[i] for i in eval_ids]
        golds = []
        for d in eval_dataset:
            golds.append(d.comment_summary)
            d.comment_summary = ["dummy"]
            d.misc["orig_comment_summary"] = "dummy"
            d.misc["comment_summary_src_ids"] = [0]

        # Perform batched queries
        tbar = tqdm(desc=f"Predicting | {prefix}")
        eval_start = time.time()
        predictions = model.batch_predict(eval_dataset, tbar=tbar, gpu_id=gpu_id)
        eval_end = time.time()
        tbar.close()

        eval_time = eval_end - eval_start

        # Save predictions & golds
        IOUtils.dump(result_dir / f"{prefix}_predictions.jsonl", predictions, IOUtils.Format.jsonList)
        IOUtils.dump(result_dir / f"{prefix}_golds.jsonl", golds, IOUtils.Format.jsonList)
        IOUtils.dump(result_dir / f"{prefix}_eval_time.json", eval_time, IOUtils.Format.json)

    def eval(self, exp_name: str, action: str = None, gpu_id: int = 0) -> None:
        if action is None:
            action = self.DEFAULT_EVAL_ACTION
        if action not in self.EVAL_ACTIONS:
            raise RuntimeError(f"Unknown eval action {action}")

        # Load eval data
        tbar = tqdm(desc="Loading data")
        dataset = MethodData.load_dataset(self.data_dir, tbar=tbar)
        indexed_dataset = {d.id: d for d in dataset}
        tbar.close()

        # Load model
        exp_dir = self.get_exp_dir(exp_name)
        model: CGModelBase = CGModelLoader.load_model(exp_dir)
        if not model.is_train_finished():
            logger.warning(f"Model not finished training, at {exp_dir}")

        # Invoke eval_one with specific data ids
        if action in [self.EVAL_VAL, self.EVAL_TESTS]:
            self.eval_one(
                exp_name,
                IOUtils.load(self.data_dir / f"split_{action}.json", IOUtils.Format.json),
                action,
                indexed_dataset,
                model,
                gpu_id=gpu_id,
            )
        elif action == self.EVAL_TESTC:
            for s1, s2 in Macros.get_pairwise_split_types_with(self.split_type):
                self.eval_one(
                    exp_name,
                    IOUtils.load(self.data_dir / f"split_{Macros.test_common}-{s1}-{s2}.json", IOUtils.Format.json),
                    f"{Macros.test_common}-{s1}-{s2}",
                    copy.deepcopy(indexed_dataset),
                    model,
                    gpu_id=gpu_id,
                )
        else:
            raise RuntimeError(f"Unknown action {action}")

    def compute_metrics_one(self, exp_name: str, prefix: str):
        # Prepare output directory
        metric_dir = self.get_metric_dir(exp_name)
        metric_dir.mkdir(parents=True, exist_ok=True)

        # Load golds and predictions
        result_dir = self.get_result_dir(exp_name)
        Utils.expect_dir_or_suggest_dvc_pull(result_dir)
        golds = IOUtils.load(result_dir / f"{prefix}_golds.jsonl", IOUtils.Format.jsonList)
        predictions = IOUtils.load(result_dir / f"{prefix}_predictions.jsonl", IOUtils.Format.jsonList)

        metrics_list: Dict[str, List] = collections.defaultdict(list)
        metrics_list["exact_match"] = EvalMetrics.batch_exact_match(golds, predictions)
        metrics_list["token_acc"] = EvalMetrics.batch_token_acc(golds, predictions)
        metrics_list["bleu"] = EvalMetrics.batch_bleu(golds, predictions)
        rouge_l_res = EvalMetrics.batch_rouge_l(golds, predictions)
        metrics_list["rouge_l_f"] = [x["f"] for x in rouge_l_res]
        metrics_list["rouge_l_p"] = [x["p"] for x in rouge_l_res]
        metrics_list["rouge_l_r"] = [x["r"] for x in rouge_l_res]
        metrics_list["meteor"] = EvalMetrics.batch_meteor(golds, predictions)
        set_match_res = EvalMetrics.batch_set_match(golds, predictions)
        metrics_list["set_match_f"] = [x["f"] for x in set_match_res]
        metrics_list["set_match_p"] = [x["p"] for x in set_match_res]
        metrics_list["set_match_r"] = [x["r"] for x in set_match_res]

        # Take average
        metrics = {}
        for k, l in metrics_list.items():
            metrics[k] = np.mean(l).item()

        # Save metrics
        IOUtils.dump(metric_dir / f"{prefix}_metrics.json", metrics, IOUtils.Format.jsonNoSort)
        IOUtils.dump(metric_dir / f"{prefix}_metrics.txt", [f"{k}: {v}" for k, v in metrics.items()], IOUtils.Format.txtList)
        IOUtils.dump(metric_dir / f"{prefix}_metrics_list.pkl", metrics_list, IOUtils.Format.pkl)

    def compute_metrics(self, exp_name: str, action: str = None) -> None:
        if action is None:
            action = self.DEFAULT_EVAL_ACTION
        if action not in self.EVAL_ACTIONS:
            raise RuntimeError(f"Unknown eval action {action}")

        if action in [self.EVAL_VAL, self.EVAL_TESTS]:
            self.compute_metrics_one(
                exp_name,
                action,
            )
        elif action == self.EVAL_TESTC:
            for s1, s2 in Macros.get_pairwise_split_types_with(self.split_type):
                self.compute_metrics_one(
                    exp_name,
                    f"{Macros.test_common}-{s1}-{s2}",
                )
        else:
            raise RuntimeError(f"Unknown action {action}")
