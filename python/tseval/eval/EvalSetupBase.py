import abc
from pathlib import Path
from typing import List


class EvalSetupBase:

    def __init__(self, work_dir: Path, work_subdir: Path, setup_name: str):
        """
        All setups require the work_dir parameter, which is configured by EvalHelper.
        Any other parameters for the implemented setup should be passed in via constructor.

        data_dir is an unit directory managed by dvc.
        """
        self.work_dir = work_dir
        self.work_subdir = work_subdir
        self.setup_name = setup_name
        return

    @property
    def setup_dir(self):
        return self.work_subdir / "setup" / self.setup_name

    @property
    def shared_data_dir(self):
        return self.work_dir / "shared"

    def get_split_dir(self, split_name: str):
        return self.work_dir / "split" / split_name

    @property
    def data_dir(self):
        return self.setup_dir / "data"

    def get_exp_dir(self, exp_name: str):
        return self.work_subdir / "exp" / self.setup_name / exp_name

    def get_result_dir(self, exp_name: str):
        return self.work_subdir / "result" / self.setup_name / exp_name

    def get_metric_dir(self, exp_name: str):
        return self.work_subdir / "metric" / self.setup_name / exp_name

    @abc.abstractmethod
    def prepare(self) -> None:
        """
        Prepares this eval setup, primarily obtains the training/validation/testing sets,
        reads data from shared_data_dir, saves any processed data to data_dir.

        setup_dir (which includes data_dir) is managed by dvc.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def train(self, exp_name: str, model_name: str, cont_train: bool, no_save: bool, **options) -> None:
        """
        Trains the model, loads data from data_dir, saves the model to
        get_exp_dir(exp_name) (intermediate files, e.g., logs, can also be saved there).

        get_exp_dir(exp_name) is managed by dvc.

        :param exp_name: name given to this experiment.
        :param model_name: the model's name.
        :param cont_train: if True and if there is already a partially trained model in
            the save directory, load that model and continue training; otherwise, ignore
            any possible partially trained models.
        :param no_save: if True, avoids saving anything during training.
        :param options: options for initializing the models.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def eval(self, exp_name: str, actions: List[str] = None, gpu_id: int = 0) -> None:
        """
        Evaluates the model (usually, on both validation and testing set),
        loads data from data_dir,
        loads trained model from get_exp_dir(exp_name),
        saves results to get_result_dir(exp_name).

        get_result_dir(exp_name) is managed by dvc.

        :param exp_name: name of the experiment.
        :param actions: a list of eval actions requested.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def compute_metrics(self, exp_name: str, actions: List[str] = None) -> None:
        """
        Computes metrics on the prediction results,
        loads data from data_dir,
        loads results from get_result_dir(exp_name),
        saves metrics to get_metric_dir(exp_name).

        get_metric_dir(exp_name) is managed by git.

        :param exp_name: name of the experiment.
        :param actions: a list of eval actions requested (to compute metrics for).
        """
        raise NotImplementedError
