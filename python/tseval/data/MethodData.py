import json
from pathlib import Path
from typing import *

from recordclass import RecordClass
from seutil import IOUtils
from tqdm import tqdm


class MethodData(RecordClass):
    id: int = -1

    # Project name
    prj: str = None
    # Years that has this data
    years: List[int] = None

    # Method name
    name: str = None
    # Code (subtokenized version after preprocessing)
    code: Union[str, List[str]] = None
    # Code with masking its name
    code_masked: str = None
    # Comment (full, including tags)
    comment: str = None
    # Comment (summary first sentence) (subtokenized version after preprocessing)
    comment_summary: Union[str, List[str]] = None
    # Class name
    cname: str = None
    # Qualified class name
    qcname: str = None
    # File relative path
    path: str = None
    # Return type
    ret: str = None
    # Parameter types
    params: List[Tuple[str, str]] = None

    misc: dict = None

    def init(self):
        self.years = []
        self.params = []
        self.misc = {}
        return

    def fill_none(self):
        if self.years is None:
            self.years = []
        if self.params is None:
            self.params = []
        if self.misc is None:
            self.misc = {}

    @classmethod
    def save_dataset(
            cls,
            dataset: List["MethodData"],
            save_dir: Path,
            exist_ok: bool = True,
            append: bool = False,
            only: Optional[Iterable[str]] = None,
    ):
        """
        Saves dataset to save_dir. Different fields are saved in different files in the
        directory.  Call graphs are shared for data from one project.
        :param dataset: the list of data to save.
        :param save_dir: the path to save.
        :param exist_ok: if False, requires that save_dir doesn't exist; otherwise,
            existing files in save_dir will be modified.
        :param append: if True, append to current saved data (requires exist_ok=True);
            otherwise, wipes out existing data at save_dir.
        :param only: only save certain fields; the files corresponding to the other fields
            are not touched; id are always saved.
        """
        save_dir.mkdir(parents=True, exist_ok=exist_ok)

        IOUtils.dump(save_dir / "id.jsonl", [d.id for d in dataset], IOUtils.Format.jsonList, append=append)

        if only is None or "prj" in only:
            IOUtils.dump(save_dir / "prj.jsonl", [d.prj for d in dataset], IOUtils.Format.jsonList, append=append)

        if only is None or "years" in only:
            IOUtils.dump(save_dir / "years.jsonl", [d.years for d in dataset], IOUtils.Format.jsonList, append=append)

        if only is None or "name" in only:
            IOUtils.dump(save_dir / "name.jsonl", [d.name for d in dataset], IOUtils.Format.jsonList, append=append)

        if only is None or "code" in only:
            IOUtils.dump(save_dir / "code.jsonl", [d.code for d in dataset], IOUtils.Format.jsonList, append=append)

        if only is None or "code_masked" in only:
            IOUtils.dump(save_dir / "code_masked.jsonl", [d.code_masked for d in dataset], IOUtils.Format.jsonList, append=append)

        if only is None or "comment" in only:
            IOUtils.dump(save_dir / "comment.jsonl", [d.comment for d in dataset], IOUtils.Format.jsonList, append=append)

        if only is None or "comment_summary" in only:
            IOUtils.dump(save_dir / "comment_summary.jsonl", [d.comment_summary for d in dataset], IOUtils.Format.jsonList, append=append)

        if only is None or "cname" in only:
            IOUtils.dump(save_dir / "cname.jsonl", [d.cname for d in dataset], IOUtils.Format.jsonList, append=append)

        if only is None or "qcname" in only:
            IOUtils.dump(save_dir / "qcname.jsonl", [d.qcname for d in dataset], IOUtils.Format.jsonList, append=append)

        if only is None or "path" in only:
            IOUtils.dump(save_dir / "path.jsonl", [d.path for d in dataset], IOUtils.Format.jsonList, append=append)

        if only is None or "ret" in only:
            IOUtils.dump(save_dir / "ret.jsonl", [d.ret for d in dataset], IOUtils.Format.jsonList, append=append)

        if only is None or "params" in only:
            IOUtils.dump(save_dir / "params.jsonl", [d.params for d in dataset], IOUtils.Format.jsonList, append=append)

        if only is None or "misc" in only:
            IOUtils.dump(save_dir / "misc.jsonl", [d.misc for d in dataset], IOUtils.Format.jsonList, append=append)

    @classmethod
    def iter_load_dataset(
            cls,
            save_dir: Path,
            only: Optional[Iterable[str]] = None,
    ) -> Generator["MethodData", None, None]:
        """
        Iteratively loads dataset from the save directory.
        :param save_dir: the directory to load data from.
        :param only: only load certain fields; the other fields are not filled in the
            loaded data; id is always loaded.
        :return: a generator iteratively loading the dataset.
        """
        if not save_dir.is_dir():
            raise FileNotFoundError(f"Not found saved data at {save_dir}")

        # First, load all ids
        ids = IOUtils.load(save_dir / "id.jsonl", IOUtils.Format.jsonList)

        # The types of some line-by-line loaded fields
        f2type = {}
        f2file = {}
        for f in ["prj", "years", "name", "code", "code_masked", "comment", "comment_summary", "cname", "qcname", "path", "ret", "params", "misc"]:
            if only is None or f in only:
                f2file[f] = open(save_dir / f"{f}.jsonl", "r")

        for i in ids:
            d = MethodData(id=i)

            # Load line-by-line fields
            for f in f2file.keys():
                o = json.loads(f2file[f].readline())
                if f in f2type:
                    o = IOUtils.dejsonfy(o, f2type[f])
                setattr(d, f, o)

            yield d

        # Close all files
        for file in f2file.values():
            file.close()

    @classmethod
    def load_dataset(
            cls,
            save_dir: Path,
            only: Optional[List[str]] = None,
            expected_ids: List[int] = None,
            tbar: Optional[tqdm] = None,
    ) -> List["MethodData"]:
        """
        Loads the dataset from save_dir.

        :param expected_ids: if provided, the list of data ids to load; the returned dataset
            will only contain these data.
        :param tbar: an optional progress bar.
        Other parameters are the same as #iter_load_dataset.
        """
        dataset = []

        # Load all data by default
        if expected_ids is None:
            expected_ids = IOUtils.load(save_dir / "id.jsonl", IOUtils.Format.jsonList)

        # Convert to set to speed up checking "has" relation
        expected_ids = set(expected_ids)

        if tbar is not None:
            tbar.set_description("Loading dataset")
            tbar.reset(len(expected_ids))

        for d in cls.iter_load_dataset(save_dir=save_dir, only=only):
            if d.id in expected_ids:
                dataset.append(d)
                if tbar is not None:
                    tbar.update(1)

                # Early stop loading if all data have been loaded
                if len(dataset) == len(expected_ids):
                    break

        return dataset
