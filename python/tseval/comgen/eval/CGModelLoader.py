from pathlib import Path

from seutil import IOUtils, LoggingUtils

from tseval.comgen.model import get_model_cls
from tseval.comgen.model.CGModelBase import CGModelBase
from tseval.Utils import Utils

logger = LoggingUtils.get_logger(__name__)


class CGModelLoader:

    @classmethod
    def init_or_load_model(
            cls,
            model_name: str,
            exp_dir: Path,
            cont_train: bool,
            no_save: bool,
            cmd_options: dict,
    ) -> CGModelBase:
        model_cls = get_model_cls(model_name)
        model_work_dir = exp_dir / "model"

        if cont_train and model_work_dir.is_dir() and not no_save:
            # Restore model name
            loaded_model_name = IOUtils.load(exp_dir / "model_name.txt", IOUtils.Format.txt)
            if model_name != loaded_model_name:
                raise ValueError(f"Contradicting model name (saved: {model_name}; new {loaded_model_name})")

            # Warning about any additional command line arguments
            if len(cmd_options) > 0:
                logger.warning(f"These options will not be used in cont_train mode: {cmd_options}")

            # Load existing model
            model: CGModelBase = model_cls.load(model_work_dir)
        else:

            if not no_save:
                exp_dir.mkdir(parents=True, exist_ok=True)

                # Save model name
                IOUtils.dump(exp_dir / "model_name.txt", model_name, IOUtils.Format.txt)

                # Prepare directory for model
                IOUtils.rm(model_work_dir)
                model_work_dir.mkdir(parents=True)

            # Initialize the model, using command line arguments
            model_options, unk_options, missing_options = Utils.parse_cmd_options_for_type(
                cmd_options,
                model_cls,
                ["self", "model_work_dir"],
            )
            if len(missing_options) > 0:
                raise KeyError(f"Missing options: {missing_options}")
            if len(unk_options) > 0:
                logger.warning(f"Unrecognized options: {unk_options}")

            model: CGModelBase = model_cls(model_work_dir=model_work_dir, no_save=no_save, **model_options)

            if not no_save:
                # Save model configs
                IOUtils.dump(exp_dir / "model_config.json", model_options, IOUtils.Format.jsonNoSort)
        return model

    @classmethod
    def load_model(cls, exp_dir: Path) -> CGModelBase:
        """
        Loads a trained model from exp_dir. Gets the model name from train_config.json.
        """
        Utils.expect_dir_or_suggest_dvc_pull(exp_dir)
        model_name = IOUtils.load(exp_dir / "model_name.txt", IOUtils.Format.txt)
        model_cls = get_model_cls(model_name)
        model_dir = exp_dir / "model"
        return model_cls.load(model_dir)
