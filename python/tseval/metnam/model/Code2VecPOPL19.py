import os
import stat
import tempfile
from pathlib import Path
from subprocess import TimeoutExpired
from typing import List, Optional

from recordclass import RecordClass
from seutil import BashUtils, IOUtils, LoggingUtils
from tqdm import tqdm

from tseval.data.MethodData import MethodData
from tseval.Environment import Environment
from tseval.Macros import Macros
from tseval.metnam.model.MNModelBase import MNModelBase
from tseval.util.ModelUtils import ModelUtils
from tseval.util.TrainConfig import TrainConfig
from tseval.Utils import Utils

logger = LoggingUtils.get_logger(__name__)


class Code2VecPOPL19Config(RecordClass):
    seed: int = None
    max_path_length: int = 8
    max_path_width: int = 2
    max_contexts: int = 200
    word_vocab_size: int = 150000
    path_vocab_size: int = 150000
    target_vocab_size: int = 25000


class Code2VecPOPL19(MNModelBase):
    ENV_NAME = "tseval-MN-Code2VecPOPL19"
    SRC_DIR = Macros.work_dir / "src" / "MN-Code2VecPOPL19"
    EXTRACTOR_RELDIR = "JavaExtractor/JPredict/target/JavaExtractor-0.0.1-SNAPSHOT.jar"

    @classmethod
    def prepare_env(cls):
        Utils.expect_dir_or_suggest_dvc_pull(cls.SRC_DIR)
        s = "#!/bin/bash\n"
        s += "set -e\n"
        s += f"source {Environment.conda_init_path}\n"
        s += f"conda env remove --name {cls.ENV_NAME}\n"
        s += f"conda create --name {cls.ENV_NAME} python=3.6 pip -y\n"
        s += f"conda activate {cls.ENV_NAME}\n"
        s += "set -x\n"
        s += f"cd {cls.SRC_DIR}\n"
        if Environment.cuda_version == "cpu":
            s += f"pip install tensorflow==2.0.0\n"
        else:
            s += f"conda install cudatoolkit=10.0 cudnn=7.6.5 -y\n"
            s += f"pip install tensorflow-gpu==2.0.0\n"
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
            seed: int = ModelUtils.get_random_seed(),
            max_path_length: int = 8,
            max_path_width: int = 2,
            max_contexts: int = 200,
            word_vocab_size: int = 150000,
            path_vocab_size: int = 150000,
            target_vocab_size: int = 25000,
    ):
        super().__init__(model_work_dir, no_save)
        if not self.SRC_DIR.is_dir():
            raise RuntimeError(f"Environment missing (expected at {self.SRC_DIR})")

        self.config = Code2VecPOPL19Config(
            seed=seed,
            max_path_length=max_path_length,
            max_path_width=max_path_width,
            max_contexts=max_contexts,
            word_vocab_size=word_vocab_size,
            path_vocab_size=path_vocab_size,
            target_vocab_size=target_vocab_size,
        )
        self.train_finished = False
        self.train_data_prepared = False

    def output_source(self, dataset: List[MethodData], output_dir: Path):
        batch_size = 500
        beg = 0
        i = 0
        while beg < len(dataset):
            batch = dataset[beg:beg+batch_size]
            cname = f"C{i:0>3d}"
            i += 1
            # Use interface to allow all modifiers (including default) on methods
            s = f"interface {cname} {{\n"
            for d in batch:
                s += d.misc["orig_code"] + "\n\n"
            s += "}\n"
            IOUtils.dump(output_dir / f"{cname}.java", s, IOUtils.Format.txt)

            beg += batch_size

    def prepare_train_val_data(
            self,
            train_dataset: List[MethodData],
            val_dataset: List[MethodData],
            data_dir: Path,
    ):
        # prepare
        train_source_dir = Path(tempfile.mkdtemp(prefix="tseval"))
        self.output_source(train_dataset, train_source_dir)
        val_source_dir = Path(tempfile.mkdtemp(prefix="tseval"))
        self.output_source(val_dataset, val_source_dir)
        train_data_file = data_dir / "data.train.raw.txt"
        val_data_file = data_dir / "data.val.raw.txt"
        tgt_hist_file = data_dir / "data.histo.tgt.c2v"
        origin_hist_file = data_dir / "data.histo.ori.c2v"
        path_hist_file = data_dir / "data.histo.path.c2v"
        error_log = data_dir / "error_log.txt"
        IOUtils.rm(error_log)

        # process
        s = "#!/bin/bash\n"
        s += "set -e\n"
        s += f"source {Environment.conda_init_path}\n"
        s += f"conda activate {self.ENV_NAME}\n"
        s += "set -x\n"
        s += f"cd {self.SRC_DIR}\n"
        s += f"python JavaExtractor/extract.py --dir {val_source_dir}" \
             f" --max_path_length {self.config.max_path_length}" \
             f" --max_path_width {self.config.max_path_width}" \
             f" --num_threads {os.cpu_count()}" \
             f" --jar {self.EXTRACTOR_RELDIR}" \
             f" > {val_data_file} 2>> {error_log}\n"
        s += f"python JavaExtractor/extract.py --dir {train_source_dir}" \
             f" --max_path_length {self.config.max_path_length}" \
             f" --max_path_width {self.config.max_path_width}" \
             f" --num_threads {os.cpu_count()}" \
             f" --jar {self.EXTRACTOR_RELDIR}" \
             f" > {train_data_file} 2>> {error_log}\n"
        # Creating histograms from the training data
        s += f"cat {train_data_file} | cut -d' ' -f1 | awk '{{n[$0]++}} END {{for (i in n) print i,n[i]}}' > {tgt_hist_file}\n"
        s += f"cat {train_data_file} | cut -d' ' -f2- | tr ' ' '\\n' | cut -d',' -f1,3 | tr ',' '\\n' | awk '{{n[$0]++}} END {{for (i in n) print i,n[i]}}' > {origin_hist_file}\n"
        s += f"cat {train_data_file} | cut -d' ' -f2- | tr ' ' '\\n' | cut -d',' -f2 | awk '{{n[$0]++}} END {{for (i in n) print i,n[i]}}' > {path_hist_file}\n"
        s += f"python preprocess.py --train_data {train_data_file} --val_data {val_data_file}" \
             f" --max_contexts {self.config.max_contexts}" \
             f" --word_vocab_size {self.config.word_vocab_size}" \
             f" --path_vocab_size {self.config.path_vocab_size}" \
             f" --target_vocab_size {self.config.target_vocab_size}" \
             f" --word_histogram {origin_hist_file}" \
             f" --path_histogram {path_hist_file}" \
             f" --target_histogram {tgt_hist_file}" \
             f" --output_name {data_dir}/data\n"
        script_path = Path(tempfile.mktemp(prefix="tseval"))
        IOUtils.dump(script_path, s, IOUtils.Format.txt)
        script_path.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
        BashUtils.run(str(script_path), 0)
        if error_log.exists():
            errors = IOUtils.load(error_log, IOUtils.Format.txt).strip()
            if len(errors) > 0:
                logger.warning(f"There are errors during data processing!\n{errors}")

        # delete temp files
        IOUtils.rm_dir(train_source_dir)
        IOUtils.rm(train_data_file)
        IOUtils.rm_dir(val_source_dir)
        IOUtils.rm(val_data_file)

    def prepare_eval_data(
            self,
            eval_dataset: List[MethodData],
            data_dir: Path,
            train_data_dir: Path,
    ):
        # prepare
        eval_source_dir = Path(tempfile.mkdtemp(prefix="tseval"))
        self.output_source(eval_dataset, eval_source_dir)
        eval_data_file = data_dir / "data.eval.raw.txt"
        tgt_hist_file = train_data_dir / "data.histo.tgt.c2v"
        origin_hist_file = train_data_dir / "data.histo.ori.c2v"
        path_hist_file = train_data_dir / "data.histo.path.c2v"
        error_log = data_dir / "error_log.txt"
        IOUtils.rm(error_log)

        # process
        s = "#!/bin/bash\n"
        s += "set -e\n"
        s += f"source {Environment.conda_init_path}\n"
        s += f"conda activate {self.ENV_NAME}\n"
        s += "set -x\n"
        s += f"cd {self.SRC_DIR}\n"
        s += f"python JavaExtractor/extract.py --dir {eval_source_dir}" \
             f" --max_path_length {self.config.max_path_length}" \
             f" --max_path_width {self.config.max_path_width}" \
             f" --num_threads {os.cpu_count()}" \
             f" --jar {self.EXTRACTOR_RELDIR}" \
             f" > {eval_data_file} 2>> {error_log}\n"
        s += f"python preprocess.py --test_data {eval_data_file}" \
             f" --max_contexts {self.config.max_contexts}" \
             f" --word_vocab_size {self.config.word_vocab_size}" \
             f" --path_vocab_size {self.config.path_vocab_size}" \
             f" --target_vocab_size {self.config.target_vocab_size}" \
             f" --word_histogram {origin_hist_file}" \
             f" --path_histogram {path_hist_file}" \
             f" --target_histogram {tgt_hist_file}" \
             f" --output_name {data_dir}/data\n"
        script_path = Path(tempfile.mktemp(prefix="tseval"))
        IOUtils.dump(script_path, s, IOUtils.Format.txt)
        script_path.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
        BashUtils.run(str(script_path), 0)
        if error_log.exists():
            errors = IOUtils.load(error_log, IOUtils.Format.txt).strip()
            if len(errors) > 0:
                logger.warning(f"There are errors during data processing!\n{errors}")

        # delete temp files
        IOUtils.rm_dir(eval_source_dir)
        IOUtils.rm(eval_data_file)

    def train(
            self,
            train_dataset: List[MethodData],
            val_dataset: List[MethodData],
            resources_path: Optional[Path] = None,
            train_config: TrainConfig = None,
    ):
        if train_config is None:
            train_config = TrainConfig()

        data_dir = self.model_work_dir / "data"
        if not self.train_data_prepared:
            IOUtils.rm_dir(data_dir)
            data_dir.mkdir(parents=True)
            self.prepare_train_val_data(train_dataset, val_dataset, data_dir)
            self.train_data_prepared = True

        model_dir = self.model_work_dir / "model"
        model_dir.mkdir(parents=True, exist_ok=True)

        # Prepare script
        s = "#!/bin/bash\n"
        s += "set -e\n"
        s += f"source {Environment.conda_init_path}\n"
        s += f"conda activate {self.ENV_NAME}\n"
        s += "set -x\n"
        s += f"cd {self.SRC_DIR}\n"
        if (model_dir / "checkpoint").exists():
            checkpoint_fcontent = IOUtils.load(model_dir / "checkpoint", IOUtils.Format.txt)
            checkpoint = None
            for line in checkpoint_fcontent.splitlines():
                k, v = line.split(":", 1)
                if k == "model_checkpoint_path":
                    checkpoint = v[:-1].split("/")[-1]
            if checkpoint is None:
                raise RuntimeError("Unable to get a checkpoint")
            s += f"CUDA_VISIBLE_DEVICES={train_config.gpu_id} timeout {train_config.train_session_time} python -u code2vec.py --data {data_dir}/data --test {data_dir}/data.val.c2v --save {model_dir}/model --load {model_dir}/{checkpoint}\n"
        else:
            s += f"CUDA_VISIBLE_DEVICES={train_config.gpu_id} timeout {train_config.train_session_time} python -u code2vec.py --data {data_dir}/data --test {data_dir}/data.val.c2v --save {model_dir}/model\n"

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

    def predict(self, data: MethodData, gpu_id: int = 0) -> List[str]:
        return self.batch_predict([data])[0]

    def batch_predict(
            self,
            dataset: List[MethodData],
            tbar: Optional[tqdm] = None,
            gpu_id: int = 0,
    ) -> List[List[str]]:
        # Figure out which checkpoint to load
        model_dir = self.model_work_dir / "model"
        checkpoint_fcontent = IOUtils.load(model_dir / "checkpoint", IOUtils.Format.txt)
        checkpoint = None
        for line in checkpoint_fcontent.splitlines():
            k, v = line.split(":", 1)
            if k == "model_checkpoint_path":
                checkpoint = v[:-1].split("/")[-1]
        if checkpoint is None:
            raise RuntimeError("Unable to get a checkpoint")

        # Prepare data
        data_dir = Path(tempfile.mkdtemp(prefix="tseval"))
        self.prepare_eval_data(dataset, data_dir, self.model_work_dir / "data")

        # Prepare script
        s = "#!/bin/bash\n"
        s += "set -e\n"
        s += f"source {Environment.conda_init_path}\n"
        s += f"conda activate {self.ENV_NAME}\n"
        s += "set -x\n"
        s += f"cd {self.SRC_DIR}\n"
        s += f"CUDA_VISIBLE_DEVICES={gpu_id} python -u code2vec.py --load {model_dir}/{checkpoint} --test {data_dir}/data.test.c2v --logs-path {model_dir}/log.txt\n"
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
        res = IOUtils.load(model_dir / "log.txt", IOUtils.Format.txt)
        decode_res = []
        for line in res.splitlines():
            decode_res.append(line.strip().split("|"))

        # Clean up
        IOUtils.rm_dir(data_dir)
        IOUtils.rm(script_path)
        IOUtils.rm(stdout_path)
        IOUtils.rm(stderr_path)

        return decode_res

    def save(self) -> None:
        # Save config and training status
        IOUtils.dump(self.model_work_dir / "config.json", IOUtils.jsonfy(self.config), IOUtils.Format.jsonPretty)
        IOUtils.dump(self.model_work_dir / "train_finished.json", self.train_finished)
        IOUtils.dump(self.model_work_dir / "train_data_prepared.json", self.train_data_prepared)

        # Model should already be saved/checkpointed to the correct path
        return

    @classmethod
    def load(cls, model_work_dir) -> "MNModelBase":
        obj = Code2VecPOPL19(model_work_dir)
        obj.config = IOUtils.dejsonfy(IOUtils.load(model_work_dir / "config.json"), Code2VecPOPL19Config)
        if (model_work_dir / "train_finished.json").exists:
            obj.train_finished = IOUtils.load(model_work_dir / "train_finished.json")
        if (model_work_dir / "train_data_prepared.json").exists:
            obj.train_data_prepared = IOUtils.load(model_work_dir / "train_data_prepared.json")
        return obj
