from typing import *

import os
from pathlib import Path


class Macros:
    this_dir: Path = Path(os.path.dirname(os.path.realpath(__file__)))
    python_dir: Path = this_dir.parent
    project_dir: Path = python_dir.parent
    paper_dir: Path = project_dir / "papers" / "acl22"

    collector_dir: Path = project_dir / "collector"
    collector_version = "0.1-dev"

    results_dir: Path = project_dir / "results"
    raw_data_dir: Path = project_dir / "_raw_data"
    work_dir: Path = project_dir / "_work"
    repos_downloads_dir: Path = project_dir / "_downloads"

    train = "train"
    val = "val"
    test = "test"
    test_common = "test_common"
    test_standard = "test_standard"

    mixed_prj = "MP"
    cross_prj = "CP"
    temporally = "T"
    split_types = [mixed_prj, cross_prj, temporally]

    @classmethod
    def get_pairwise_split_types_with(cls, split: str) -> List[Tuple[str, str]]:
        pairs = []
        before = True
        for s in cls.split_types:
            if s == split:
                before = False
            else:
                if before:
                    pairs.append((s, split))
                else:
                    pairs.append((split, s))
        return pairs

    com_gen = "CG"
    met_nam = "MN"

    tasks = ["CG", "MN"]
