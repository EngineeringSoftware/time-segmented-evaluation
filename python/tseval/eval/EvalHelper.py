import collections
import copy
import math
import random
from typing import Dict, List, Tuple

from seutil import IOUtils

from tseval.data.MethodData import MethodData
from tseval.Macros import Macros
from tseval.Utils import Utils


class EvalHelper:

    def get_splits(
            self,
            split_name: str,
            seed: int,
            prj_val_ratio: float = 0.1,
            prj_test_ratio: float = 0.2,
            inprj_val_ratio: float = 0.1,
            inprj_test_ratio: float = 0.2,
            train_year: int = 2019,
            val_year: int = 2020,
            test_year: int = 2021,
            debug: bool = False,
    ):
        """
        Gets {mixed-project, cross-project, temporally} splits for the given seed and configurations.

        :param debug: take maximum of 500/100/100 data in the result train/val/test sets.
        """
        split_dir = Macros.work_dir / "split" / split_name
        IOUtils.rm_dir(split_dir)
        split_dir.mkdir(parents=True)

        # Save configs
        IOUtils.dump(
            split_dir / "config.json",
            {
                "seed": seed,
                "prj_val_ratio": prj_val_ratio,
                "prj_test_ratio": prj_test_ratio,
                "inprj_val_ratio": inprj_val_ratio,
                "inprj_test_ratio": inprj_test_ratio,
                "train_year": train_year,
                "val_year": val_year,
                "test_year": test_year,
                "debug": debug,
            },
            IOUtils.Format.jsonNoSort,
        )
        stats = {}

        # Load shared data
        dataset: List[MethodData] = MethodData.load_dataset(Macros.work_dir / "shared")

        # Initialize random state
        random.seed(seed)

        all_prjs = list(sorted(set([d.prj for d in dataset])))
        prj2ids = collections.defaultdict(list)
        for d in dataset:
            prj2ids[d.prj].append(d.id)

        # Get project split
        prj_split_names: Dict[str, List[str]] = {sn: l for sn, l in zip(
            [Macros.train, Macros.val, Macros.test],
            self.split(all_prjs, prj_val_ratio, prj_test_ratio),
        )}
        prj_split: Dict[str, List[int]] = {sn: sum([prj2ids[n] for n in names], [])
                                            for sn, names in prj_split_names.items()}
        for sn in [Macros.train, Macros.val, Macros.test]:
            IOUtils.dump(split_dir / f"prj-{sn}.jsonl", prj_split_names[sn], IOUtils.Format.jsonList)
            IOUtils.dump(split_dir / f"prj-split-{sn}.jsonl", prj_split[sn], IOUtils.Format.jsonList)
            stats[f"num_prj_{sn}"] = len(prj_split_names[sn])
            stats[f"num_prj_split_{sn}"] = len(prj_split[sn])

        # Get in-project splits
        prj2inprj_splits: Dict[str, Dict[str, List[str]]] = {}

        for prj in all_prjs:
            prj2inprj_splits[prj] = {sn: l for sn, l in zip(
                [Macros.train, Macros.val, Macros.test],
                self.split(prj2ids[prj], inprj_val_ratio, inprj_test_ratio),
            )}
        inprj_split: Dict[str, List[int]] = {sn: sum([prj2inprj_splits[prj][sn] for prj in all_prjs], [])
                                             for sn in [Macros.train, Macros.val, Macros.test]}
        for sn in [Macros.train, Macros.val, Macros.test]:
            IOUtils.dump(split_dir / f"inprj-split-{sn}.jsonl", inprj_split[sn], IOUtils.Format.jsonList)
            stats[f"num_inprj_split_{sn}"] = len(inprj_split[sn])

        # Get year splits
        year_split: Dict[str, List[int]] = {sn: [] for sn in [Macros.train, Macros.val, Macros.test]}
        for d in dataset:
            min_year = min(d.years)
            if min_year <= train_year:
                year_split[Macros.train].append(d.id)
            elif min_year <= val_year:
                year_split[Macros.val].append(d.id)
            elif min_year <= test_year:
                year_split[Macros.test].append(d.id)
        for sn in [Macros.train, Macros.val, Macros.test]:
            IOUtils.dump(split_dir / f"year-split-{sn}.jsonl", year_split[sn], IOUtils.Format.jsonList)
            stats[f"num_year_split_{sn}"] = len(year_split[sn])

        # Get actual mixed-prj/cross-prj/temporally splits
        train_size = min(len(inprj_split[Macros.train]), len(prj_split[Macros.train]), len(year_split[Macros.train]))
        split_sn2split_ids: Dict[Tuple[str, str], List[int]] = {}
        for sn in [Macros.train, Macros.val, Macros.test]:
            split_sn2split_ids[(Macros.mixed_prj, sn)] = list(sorted(inprj_split[sn]))
            split_sn2split_ids[(Macros.cross_prj, sn)] = list(sorted(prj_split[sn]))
            split_sn2split_ids[(Macros.temporally, sn)] = list(sorted(year_split[sn]))

        for s1_i, s1 in enumerate(Macros.split_types):
            # train/eval/test_standard
            for sn in [Macros.train, Macros.val, Macros.test]:
                split_ids = split_sn2split_ids[(s1, sn)]

                # Downsample train set
                if sn == Macros.train:
                    IOUtils.dump(split_dir / f"{s1}-{sn}_full.json", split_ids, IOUtils.Format.json)
                    random.shuffle(split_ids)
                    split_ids = list(sorted(split_ids[:train_size]))

                # Debugging
                if debug:
                    if sn == Macros.train:
                        split_ids = split_ids[:500]
                    else:
                        split_ids = split_ids[:100]

                sn_oname = sn if sn != Macros.test else Macros.test_standard
                IOUtils.dump(split_dir / f"{s1}-{sn_oname}.json", split_ids, IOUtils.Format.json)
                stats[f"num_{s1}_{sn_oname}"] = len(split_ids)

            # test_common set
            for s2_i in range(s1_i+1, len(Macros.split_types)):
                s2 = Macros.split_types[s2_i]
                split_ids = self.intersect(
                    split_sn2split_ids[(s1, Macros.test)],
                    split_sn2split_ids[(s2, Macros.test)],
                )

                # Debugging
                if debug:
                    split_ids = split_ids[:100]

                IOUtils.dump(split_dir / f"{s1}-{s2}-{Macros.test_common}.json", split_ids, IOUtils.Format.json)
                stats[f"num_{s1}-{s2}_{Macros.test_common}"] = len(split_ids)

        # Save stats
        IOUtils.dump(split_dir / "stats.json", stats, IOUtils.Format.jsonNoSort)

        # Suggest dvc command
        print(Utils.suggest_dvc_add(split_dir))
        return

    @classmethod
    def split(
            cls,
            l: List,
            val_ratio: float,
            test_ratio: float,
    ) -> Tuple[List, List, List]:
        assert val_ratio > 0 and test_ratio > 0 and val_ratio + test_ratio < 1
        lcopy = copy.copy(l)
        random.shuffle(lcopy)
        test_val_split = int(math.ceil(len(lcopy) * test_ratio))
        val_train_split = int(math.ceil(len(lcopy) * (test_ratio + val_ratio)))
        return (
            lcopy[val_train_split:],
            lcopy[test_val_split:val_train_split],
            lcopy[:test_val_split],
        )

    @classmethod
    def intersect(cls, *lists: List[int])-> List[int]:
        return list(sorted(set.intersection(*[set(l) for l in lists])))

    @classmethod
    def get_task_specific_eval_helper(cls, task):
        if task == Macros.com_gen:
            from tseval.comgen.eval.CGEvalHelper import CGEvalHelper
            return CGEvalHelper()
        elif task == Macros.met_nam:
            from tseval.metnam.eval.MNEvalHelper import MNEvalHelper
            return MNEvalHelper()
        else:
            raise KeyError(f"Invalid task {task}")

    def exp_prepare(self, **options):
        task = options.pop("task")
        eh = self.get_task_specific_eval_helper(task)
        setup_cls_name = options.pop("setup")
        setup_name = options.pop("setup_name")
        eh.exp_prepare(setup_cls_name, setup_name, **options)

    def exp_train(self, **options):
        task = options.pop("task")
        eh = self.get_task_specific_eval_helper(task)
        setup_name = options.pop("setup_name")
        exp_name = options.pop("exp_name")
        model_name = options.pop("model_name")
        cont_train = Utils.get_option_as_boolean(options, "cont_train", pop=True)
        no_save = Utils.get_option_as_boolean(options, "no_save", pop=True)
        eh.exp_train(setup_name, exp_name, model_name, cont_train, no_save, **options)

    def exp_eval(self, **options):
        task = options.pop("task")
        eh = self.get_task_specific_eval_helper(task)
        setup_name = options.pop("setup_name")
        exp_name = options.pop("exp_name")
        action = options.pop("action")
        gpu_id = Utils.get_option_and_pop(options, "gpu_id", 0)
        eh.exp_eval(setup_name, exp_name, action, gpu_id=gpu_id)

    def exp_compute_metrics(self, **options):
        task = options.pop("task")
        eh = self.get_task_specific_eval_helper(task)
        setup_name = options.pop("setup_name")
        exp_name = options.pop("exp_name")
        action = options.pop("action")
        eh.exp_compute_metrics(setup_name, exp_name, action)
