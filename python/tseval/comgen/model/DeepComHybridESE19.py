import collections
import copy
import stat
import tempfile
from pathlib import Path
from subprocess import TimeoutExpired
from typing import List, Optional

import torch
from recordclass import RecordClass
from seutil import BashUtils, IOUtils, LoggingUtils
from tqdm import tqdm

from tseval.comgen.model.CGModelBase import CGModelBase
from tseval.data.MethodData import MethodData
from tseval.Environment import Environment
from tseval.Macros import Macros
from tseval.util.ModelUtils import ModelUtils
from tseval.util.TrainConfig import TrainConfig
from tseval.Utils import Utils

logger = LoggingUtils.get_logger(__name__)


class DeepComHybridESE19Config(RecordClass):
    vocab_size: int = 50_000
    max_code_len: int = 200
    max_sbt_len: int = 500
    max_tgt_len: int = 50
    seed: int = None


class DeepComHybridESE19(CGModelBase):
    ENV_NAME = "tseval-CG-DeepComHybridESE19"
    SRC_DIR = Macros.work_dir / "src" / "CG-DeepComHybridESE19"

    _BOS = '<S>'
    _EOS = '</S>'
    _UNK = '<UNK>'
    _KEEP = '<KEEP>'
    _DEL = '<DEL>'
    _INS = '<INS>'
    _SUB = '<SUB>'
    _NONE = '<NONE>'

    _START_VOCAB = [_BOS, _EOS, _UNK, _KEEP, _DEL, _INS, _SUB, _NONE]

    BOS_ID = 0
    EOS_ID = 1
    UNK_ID = 2
    KEEP_ID = 3
    DEL_ID = 4
    INS_ID = 5
    SUB_ID = 6
    NONE_ID = 7

    @classmethod
    def prepare_env(cls):
        Utils.expect_dir_or_suggest_dvc_pull(cls.SRC_DIR)
        s = "#!/bin/bash\n"
        s += "set -e\n"
        s += f"source {Environment.conda_init_path}\n"
        s += f"conda env remove --name {cls.ENV_NAME}\n"
        s += f"conda create --name {cls.ENV_NAME} python=3.7 pip -y\n"
        s += f"conda activate {cls.ENV_NAME}\n"
        s += "set -x\n"
        s += f"cd {cls.SRC_DIR}\n"
        if Environment.cuda_version == "cpu":
            s += f"pip install tensorflow==1.15\n"
        else:
            s += f"pip install tensorflow-gpu==1.15\n"
        s += f"pip install -r requirements.txt\n"
        t = Path(tempfile.mktemp(prefix="tseval"))
        IOUtils.dump(t, s, IOUtils.Format.txt)
        t.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
        print(f"Preparing env for {__name__}...")
        rr = BashUtils.run(t)
        if rr.return_code == 0:
            print("Success!")
        else:
            print("Failed!")
            print(f"STDOUT:\n{rr.stdout}")
            print(f"STDERR:\n{rr.stderr}")
            print(f"^^^ Preparing env for {__name__} failed!")

    def __init__(
            self,
            model_work_dir: Path,
            no_save: bool = False,
            vocab_size: int = 50_000,
            max_code_len: int = 200,
            max_sbt_len: int = 500,
            max_tgt_len: int = 50,
            seed: int = ModelUtils.get_random_seed(),
    ):
        super().__init__(model_work_dir, no_save)
        if not self.SRC_DIR.is_dir():
            raise RuntimeError(f"Environment missing (expected at {self.SRC_DIR})")

        self.config = DeepComHybridESE19Config(
            vocab_size=vocab_size,
            max_code_len=max_code_len,
            max_sbt_len=max_sbt_len,
            max_tgt_len=max_tgt_len,
            seed=seed,
        )
        self.train_finished = False
        self.train_data_prepared = False

    def prepare_data(
            self,
            dataset: List[MethodData],
            data_dir: Path,
            sn: str,
            train: bool = False,
    ):
        sn_dir = data_dir / sn
        sn_dir.mkdir(parents=True, exist_ok=True)
        prefix = sn if train else "test"

        # code
        IOUtils.dump(
            sn_dir / f"{prefix}.orig.code",
            [d.misc["orig_code"].replace("\n", " ") for d in dataset],
            IOUtils.Format.txtList,
        )

        # process
        s = "#!/bin/bash\n"
        s += "set -e\n"
        s += f"source {Environment.conda_init_path}\n"
        s += f"conda activate {self.ENV_NAME}\n"
        s += "set -x\n"
        s += f"cd {self.SRC_DIR}/data_utils\n"
        s += f"python get_ast.py '{sn_dir}/{prefix}.orig.code' '{sn_dir}/{prefix}.token.code' '{sn_dir}/{prefix}.ast.json'\n"
        s += f"python ast_traversal.py '{sn_dir}/{prefix}.ast.json' '{sn_dir}/{prefix}.token.sbt'\n"
        script_path = Path(tempfile.mktemp(prefix="tseval"))
        IOUtils.dump(script_path, s, IOUtils.Format.txt)
        script_path.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
        BashUtils.run(str(script_path), 0)

        # subtokenize code
        IOUtils.dump(
            sn_dir / f"{prefix}.token.code",
            [" ".join(d.code[:self.config.max_code_len]) for d in dataset],
            IOUtils.Format.txtList,
        )

        # subtokenized comment
        IOUtils.dump(
            sn_dir / f"{prefix}.token.nl",
            [" ".join(d.comment_summary[:self.config.max_tgt_len]) for d in dataset],
            IOUtils.Format.txtList,
        )

        # init vocab
        if train:
            for x in ["code", "sbt", "nl"]:
                subtokenized = IOUtils.load(sn_dir / f"{prefix}.token.{x}", IOUtils.Format.txtList)
                vocab_counter = collections.Counter()
                for l in subtokenized:
                    for t in l.split():
                        vocab_counter[t] += 1
                vocab = copy.copy(self._START_VOCAB)
                vocab += [x for x, _ in vocab_counter.most_common()]
                vocab = vocab[:self.config.vocab_size]
                IOUtils.dump(data_dir / f"vocab.{x}", vocab, IOUtils.Format.txtList)

    def train(
            self,
            train_dataset: List[MethodData],
            val_dataset: List[MethodData],
            resources_path: Optional[Path] = None,
            train_config: Optional[TrainConfig] = None,
    ):
        if train_config is None:
            train_config = TrainConfig()

        data_dir = self.model_work_dir / "data"
        if not self.train_data_prepared:
            IOUtils.rm_dir(data_dir)
            data_dir.mkdir(parents=True)
            self.prepare_data(train_dataset, data_dir, "train", train=True)
            self.prepare_data(val_dataset, data_dir, "dev")
            self.train_data_prepared = True

        # Prepare config
        config = IOUtils.load(self.SRC_DIR / "config" / "config.yaml", IOUtils.Format.yaml)
        config["data_dir"] = str(data_dir)
        model_dir = self.model_work_dir / "model"
        config["model_dir"] = str(model_dir)
        config["dev_prefix"] = ["dev"]
        if torch.cuda.is_available():
            config["gpu_id"] = train_config.gpu_id
        else:
            config["no_gpu"] = True

        for x in config["encoders"]:
            if x["name"] == "code":
                x["max_len"] = self.config.max_code_len
            if x["name"] == "sbt":
                x["max_len"] = self.config.max_sbt_len
        for x in config["decoders"]:
            if x["name"] == "nl":
                x["max_len"] = self.config.max_tgt_len

        # Adjust checkpoint and validation to be per-epoch
        batch_size = config["batch_size"]
        step_per_epoch = max(len(train_dataset) // batch_size, 1)
        config["steps_per_checkpoint"] = step_per_epoch
        config["steps_per_eval"] = step_per_epoch

        config_path = self.model_work_dir / "DeepCom-config.yaml"
        IOUtils.dump(config_path, config, IOUtils.Format.yaml)

        # Prepare script
        s = "#!/bin/bash\n"
        s += "set -e\n"
        s += f"source {Environment.conda_init_path}\n"
        s += f"conda activate {self.ENV_NAME}\n"
        s += "set -x\n"
        s += f"cd '{self.SRC_DIR}/source code'\n"
        s += f"""timeout {train_config.train_session_time} python __main__.py {config_path} \\
            --train \\
            -v \\
            --tf-seed {self.config.seed} \\
            --seed {self.config.seed}
        """

        self.fix_checkpoint_path()

        script_path = self.model_work_dir / "train.sh"
        stdout_path = self.model_work_dir / "train.stdout"
        stderr_path = self.model_work_dir / "train.stderr"
        IOUtils.dump(script_path, s, IOUtils.Format.txt)
        script_path.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)

        with IOUtils.cd(self.model_work_dir):
            try:
                logger.info(f"=====Starting train\nScript: {script_path}\nSTDOUT: {stdout_path}\nSTDERR: {stderr_path}\n=====")
                rr = BashUtils.run(f"{script_path} 1>{stdout_path} 2>{stderr_path}", timeout=train_config.train_session_time)
                if rr.return_code != 0:
                    raise RuntimeError(f"Train returned {rr.return_code}; check STDERR at {stderr_path}")
            except (TimeoutExpired, KeyboardInterrupt):
                logger.warning("Training not finished")
                self.train_finished = False
                return
            except:
                logger.warning(f"Error during training!")
                raise

        # If we can reach here, training should be finished
        self.train_finished = True
        return

    def is_train_finished(self) -> bool:
        return self.train_finished

    def predict(
            self,
            data: MethodData,
            gpu_id: int = 0,
    ) -> List[str]:
        return self.batch_predict([data], gpu_id=gpu_id)[0]

    def batch_predict(
            self,
            dataset: List[MethodData],
            tbar: Optional[tqdm] = None,
            gpu_id: int = 0,
    ) -> List[List[str]]:
        # Prepare data
        data_dir = Path(tempfile.mkdtemp(prefix="tseval"))

        # Use the dummy comment_summary field to carry id information, so that we know what ids are deleted
        for i, d in enumerate(dataset):
            d.comment_summary = [str(i)]
        self.prepare_data(dataset, data_dir, "test", train=False)

        # Prepare config
        config = IOUtils.load(self.model_work_dir / "DeepCom-config.yaml", IOUtils.Format.yaml)
        config["data_dir"] = str(data_dir)
        config_path = Path(tempfile.mktemp(prefix="tseval"))
        config["model_dir"] = str(self.model_work_dir / "model")
        if torch.cuda.is_available():
            config["gpu_id"] = gpu_id
        else:
            config["no_gpu"] = True
        IOUtils.dump(config_path, config, IOUtils.Format.yaml)

        # Prepare script
        output_path = Path(tempfile.mktemp(prefix="tseval"))
        s = "#!/bin/bash\n"
        s += "set -e\n"
        s += f"source {Environment.conda_init_path}\n"
        s += f"conda activate {self.ENV_NAME}\n"
        s += "set -x\n"
        s += f"cd '{self.SRC_DIR}/source code'\n"
        s += f"""python __main__.py {config_path} \\
            --decode test \\
            --output {output_path}
        """

        self.fix_checkpoint_path()

        script_path = Path(tempfile.mktemp(prefix="tseval.test.sh-"))
        stdout_path = Path(tempfile.mktemp(prefix="tseval.test.stdout-"))
        stderr_path = Path(tempfile.mktemp(prefix="tseval.test.stderr-"))
        IOUtils.dump(script_path, s, IOUtils.Format.txt)
        script_path.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)

        # Run predictions
        with IOUtils.cd(self.model_work_dir):
            logger.info(f"=====Starting eval\nScript: {script_path}\nSTDOUT: {stdout_path}\nSTDERR: {stderr_path}\n=====")
            rr = BashUtils.run(f"{script_path} 1>{stdout_path} 2>{stderr_path}")
            if rr.return_code != 0:
                raise RuntimeError(f"Eval returned {rr.return_code}; check STDERR at {stderr_path}")

        # Load predictions
        decode_res = [x.split() for x in IOUtils.load(output_path, IOUtils.Format.txtList)]

        # Delete temp files
        IOUtils.rm_dir(data_dir)
        IOUtils.rm(script_path)
        IOUtils.rm(stdout_path)
        IOUtils.rm(stderr_path)
        IOUtils.rm(output_path)
        IOUtils.rm(config_path)

        return decode_res

    def save(self) -> None:
        # Save config and training status
        IOUtils.dump(self.model_work_dir / "config.json", IOUtils.jsonfy(self.config), IOUtils.Format.jsonPretty)
        IOUtils.dump(self.model_work_dir / "train_finished.json", self.train_finished)
        IOUtils.dump(self.model_work_dir / "train_data_prepared.json", self.train_data_prepared)

        # Model should already be saved/checkpointed to the correct path
        return

    @classmethod
    def load(cls, model_work_dir) -> "CGModelBase":
        obj = DeepComHybridESE19(model_work_dir)
        obj.config = IOUtils.dejsonfy(IOUtils.load(model_work_dir / "config.json"), DeepComHybridESE19Config)
        if (model_work_dir / "train_finished.json").exists:
            obj.train_finished = IOUtils.load(model_work_dir / "train_finished.json")
        if (model_work_dir / "train_data_prepared.json").exists:
            obj.train_data_prepared = IOUtils.load(model_work_dir / "train_data_prepared.json")
        return obj

    def fix_checkpoint_path(self):
        """
        Fixes the absolute path in checkpoints/checkpoint file, which would cause error
        if the model has been moved around.
        """
        checkpoint_file = self.model_work_dir / "model" / "checkpoints" / "checkpoint"
        if checkpoint_file.exists():
            checkpoint_content = IOUtils.load(checkpoint_file, IOUtils.Format.txt)
            pattern = f"/model/checkpoints/"
            fixed_checkpoint_content = ""
            for line in checkpoint_content.splitlines():
                if pattern in line:
                    old_path = line.split('"')[1].split(pattern)[0]
                    fixed_checkpoint_content += line.replace(old_path, str(self.model_work_dir)) + "\n"
                else:
                    fixed_checkpoint_content += line + "\n"
            IOUtils.dump(checkpoint_file, fixed_checkpoint_content, IOUtils.Format.txt)
