from pathlib import Path
from typing import Optional

from seutil import IOUtils, LoggingUtils

from tseval.eval.EvalSetupBase import EvalSetupBase
from tseval.Macros import Macros
from tseval.metnam.eval import get_setup_cls
from tseval.Utils import Utils

logger = LoggingUtils.get_logger(__name__)


class MNEvalHelper:

    def __init__(self):
        self.work_subdir: Path = Macros.work_dir / "MN"

    def exp_prepare(self, setup: str, setup_name: str, **cmd_options):
        # Clean the setup dir
        setup_dir: Path = self.work_subdir / "setup" / setup_name
        IOUtils.rm_dir(setup_dir)
        setup_dir.mkdir(parents=True)

        # Initialize setup
        setup_cls = get_setup_cls(setup)
        setup_options, unk_options, missing_options = Utils.parse_cmd_options_for_type(
            cmd_options,
            setup_cls,
            ["self", "work_dir", "work_subdir", "setup_name"],
        )
        if len(missing_options) > 0:
            raise KeyError(f"Missing options: {missing_options}")
        if len(unk_options) > 0:
            logger.warning(f"Unrecognized options: {unk_options}")
        setup_obj: EvalSetupBase = setup_cls(
            work_dir=Macros.work_dir,
            work_subdir=self.work_subdir,
            setup_name=setup_name,
            **setup_options,
        )

        # Save setup configs
        setup_options["setup"] = setup
        IOUtils.dump(setup_dir / "setup_config.json", setup_options, IOUtils.Format.jsonNoSort)

        # Prepare data
        setup_obj.prepare()

        # Print dvc commands
        print(Utils.suggest_dvc_add(setup_obj.setup_dir))

    def load_setup(self, setup_dir: Path, setup_name: str) -> EvalSetupBase:
        """
        Loads the setup from its save directory, with updating setup_name.
        """
        config = IOUtils.load(setup_dir / "setup_config.json", IOUtils.Format.json)
        setup_cls = get_setup_cls(config.pop("setup"))
        setup_obj = setup_cls(work_dir=Macros.work_dir, work_subdir=self.work_subdir, setup_name=setup_name, **config)
        return setup_obj

    def exp_train(
            self,
            setup_name: str,
            exp_name: str,
            model_name: str,
            cont_train: bool,
            no_save: bool,
            **cmd_options,
    ):
        # Load saved setup
        setup_dir = self.work_subdir / "setup" / setup_name
        Utils.expect_dir_or_suggest_dvc_pull(setup_dir)
        setup = self.load_setup(setup_dir, setup_name)

        if not cont_train:
            # Delete existing trained model
            IOUtils.rm_dir(setup.get_exp_dir(exp_name))

        # Invoke training
        setup.train(exp_name, model_name, cont_train, no_save, **cmd_options)

        # Print dvc commands
        print(Utils.suggest_dvc_add(setup.get_exp_dir(exp_name)))

    def exp_eval(
            self,
            setup_name: str,
            exp_name: str,
            action: Optional[str],
            gpu_id: int = 0,
    ):
        # Load saved setup
        setup_dir = self.work_subdir / "setup" / setup_name
        Utils.expect_dir_or_suggest_dvc_pull(setup_dir)
        setup = self.load_setup(setup_dir, setup_name)

        # Invoke eval
        setup.eval(exp_name, action, gpu_id=gpu_id)

        # Print dvc commands
        print(Utils.suggest_dvc_add(setup.get_result_dir(exp_name)))

    def exp_compute_metrics(
            self,
            setup_name: str,
            exp_name: str,
            action: Optional[str] = None,
    ):
        # Load saved setup
        setup_dir = self.work_subdir / "setup" / setup_name
        Utils.expect_dir_or_suggest_dvc_pull(setup_dir)
        setup = self.load_setup(setup_dir, setup_name)

        # Invoke eval
        setup.compute_metrics(exp_name, action)

        # Print dvc commands
        print(Utils.suggest_dvc_add(setup.get_metric_dir(exp_name)))
