import abc
from pathlib import Path
from typing import List, Optional, Tuple

from seutil import IOUtils
from tqdm import tqdm

from tseval.data.MethodData import MethodData
from tseval.util.TrainConfig import TrainConfig


class CGModelBase:

    def __init__(self, model_work_dir: Path, no_save: bool = False):
        self.model_work_dir = model_work_dir
        self.no_save = no_save

    @abc.abstractmethod
    def train(
            self,
            train_dataset: List[MethodData],
            val_dataset: List[MethodData],
            resources_path: Optional[Path] = None,
            train_config: Optional[TrainConfig] = None,
    ):
        """
        Trains the model.

        :param train_dataset: training set.
        :param val_dataset: validation set.
        :param resources_path: path to resources that could be shared by multiple model's training process,
            e.g., pre-trained embeddings.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def is_train_finished(self) -> bool:
        raise NotImplementedError

    @abc.abstractmethod
    def predict(
            self,
            data: MethodData,
            gpu_id: int = 0,
    ) -> List[str]:
        """
        Predicts the comment summary given the context in data. The model should output
        results with a confidence score in [0, 1].
        :param data: the data, with its statements partially filled.
        :return: a list of predicted comment summary tokens.
        """
        raise NotImplementedError

    def batch_predict(
            self,
            dataset: List[MethodData],
            tbar: Optional[tqdm] = None,
            gpu_id: int = 0,
    ) -> List[List[str]]:
        """
        Performs batched predictions using given dataset as inputs.

        The default implementation invokes #predict multiple times. Subclass can override
        this method to speed up the prediction by using batching.

        :param dataset: a list of inputs.
        :param tbar: an optional tqdm progress bar to show prediction progress.
        :return: a list of the return value of #predict.
        """
        if tbar is not None:
            tbar.reset(len(dataset))

        results = []
        for data in dataset:
            results.append(self.predict(data, gpu_id=gpu_id))
            if tbar is not None:
                tbar.update(1)

        return results

    def save(self) -> None:
        """
        Saves the current model at the work_dir.
        Default behavior is to serialize the entire object in model.pkl.
        """
        if not self.no_save:
            IOUtils.dump(self.model_work_dir / "model.pkl", self, IOUtils.Format.pkl)

    @classmethod
    def load(cls, work_dir) -> "CGModelBase":
        """
        Loads a model from the work_dir.
        Default behavior is to deserialize the object from model.pkl, with resetting its work_dir.
        """
        obj = IOUtils.load(work_dir / "model.pkl", IOUtils.Format.pkl)
        obj.model_work_dir = work_dir
        return obj
