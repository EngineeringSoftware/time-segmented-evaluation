import stat
import tempfile
from pathlib import Path
from subprocess import TimeoutExpired
from typing import List, Optional

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


class TransformerACL20Config(RecordClass):
    max_src_len: int = 150
    max_tgt_len: int = 50
    use_rnn: bool = False
    seed: int = None


class TransformerACL20(CGModelBase):
    ENV_NAME = "tseval-CG-TransformerACL20"
    SRC_DIR = Macros.work_dir / "src" / "CG-TransformerACL20"

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
        # Pytorch 1.4.0 -> max cuda version 10.1
        cuda_toolkit_spec = Environment.get_cuda_toolkit_spec()
        if cuda_toolkit_spec in ["cudatoolkit=11.0", "cudatoolkit=10.2"]:
            cuda_toolkit_spec = "cudatoolkit=10.1"
        s += f"conda install -y pytorch==1.4.0 torchvision==0.5.0 {cuda_toolkit_spec} -c pytorch\n"
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
            max_src_len: int = 150,
            max_tgt_len: int = 50,
            use_rnn: bool = False,
            seed: int = ModelUtils.get_random_seed(),
    ):
        super().__init__(model_work_dir, no_save)
        if not self.SRC_DIR.is_dir():
            raise RuntimeError(f"Environment missing (expected at {self.SRC_DIR})")

        self.config = TransformerACL20Config(
            max_src_len=max_src_len,
            max_tgt_len=max_tgt_len,
            use_rnn=use_rnn,
            seed=seed,
        )
        self.train_finished = False

    def prepare_data(self, dataset: List[MethodData], data_dir: Path, sn: str):
        sn_dir = data_dir / "java" / sn
        IOUtils.rm_dir(sn_dir)
        sn_dir.mkdir(parents=True)

        # subtokenized code
        IOUtils.dump(
            sn_dir / "code.original_subtoken",
            [" ".join(d.code[:self.config.max_src_len]) for d in dataset],
            IOUtils.Format.txtList,
        )

        # subtokenized comment
        IOUtils.dump(
            sn_dir / "javadoc.original",
            [" ".join(d.comment_summary[:self.config.max_tgt_len]) for d in dataset],
            IOUtils.Format.txtList,
        )

        # tokenized code
        with open(sn_dir / "code.original", "w") as f:
            for d in dataset:
                tokens = ModelUtils.regroup_subtokens(d.code, d.misc["code_src_ids"])
                f.write(" ".join(tokens) + "\n")

    def train(
            self,
            train_dataset: List[MethodData],
            val_dataset: List[MethodData],
            resources_path: Optional[Path] = None,
            train_config: TrainConfig = None,
    ):
        if train_config is None:
            train_config = TrainConfig()

        # Prepare data
        data_dir = self.model_work_dir / "data"
        if not data_dir.is_dir():
            data_dir.mkdir(parents=True)
            self.prepare_data(train_dataset, data_dir, "train")
            self.prepare_data(val_dataset, data_dir, "dev")

        # Prepare script
        model_dir = self.model_work_dir / "model"
        s = "#!/bin/bash\n"
        s += "set -e\n"
        s += f"source {Environment.conda_init_path}\n"
        s += f"conda activate {self.ENV_NAME}\n"
        s += "set -x\n"
        s += f"cd {self.SRC_DIR}\n"
        s += f"""MKL_THREADING_LAYER=GNU CUDA_VISIBLE_DEVICES={train_config.gpu_id} timeout {train_config.train_session_time} python -W ignore '{self.SRC_DIR}/main/train.py' \\
               --data_workers 5 \\
               --dataset_name java \\
               --data_dir '{data_dir}/' \\
               --model_dir '{model_dir}' \\
               --model_name model \\
               --train_src train/code.original_subtoken \\
               --train_tgt train/javadoc.original \\
               --dev_src dev/code.original_subtoken \\
               --dev_tgt dev/javadoc.original \\
               --uncase True \\
               --use_src_word True \\
               --use_src_char False \\
               --use_tgt_word True \\
               --use_tgt_char False \\
               --max_src_len {self.config.max_src_len} \\
               --max_tgt_len {self.config.max_tgt_len} \\
               --emsize 512 \\
               --fix_embeddings False \\
               --src_vocab_size 50000 \\
               --tgt_vocab_size 30000 \\
               --share_decoder_embeddings True \\
               --max_examples -1 \\
               --batch_size 32 \\
               --test_batch_size 64 \\
               --num_epochs 200 \\
               --dropout_emb 0.2 \\
               --dropout 0.2 \\
               --copy_attn True \\
               --early_stop 20 \\
               --optimizer adam \\
               --lr_decay 0.99 \\
               --valid_metric bleu \\
               --checkpoint True \\
               --random_seed {self.config.seed} \\
            """

        if not self.config.use_rnn:
            s += """--model_type transformer \\
               --num_head 8 \\
               --d_k 64 \\
               --d_v 64 \\
               --d_ff 2048 \\
               --src_pos_emb False \\
               --tgt_pos_emb True \\
               --max_relative_pos 32 \\
               --use_neg_dist True \\
               --nlayers 6 \\
               --trans_drop 0.2 \\
               --warmup_steps 2000 \\
               --learning_rate 0.0001
            """
        else:
            s += """--model_type rnn \\
               --conditional_decoding False \\
               --nhid 512 \\
               --nlayers 2 \\
               --use_all_enc_layers False \\
               --dropout_rnn 0.2 \\
               --reuse_copy_attn True \\
               --learning_rate 0.002 \\
               --grad_clipping 5.0
            """

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
        # Remove the big checkpoint file
        IOUtils.rm(model_dir / "model.mdl.checkpoint")
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
        # Prepare data
        data_dir = Path(tempfile.mkdtemp(prefix="tseval"))

        # Use the dummy comment_summary field to carry id information, so that we know what ids are deleted
        for i, d in enumerate(dataset):
            d.comment_summary = [str(i)]
        self.prepare_data(dataset, data_dir, "test")

        # Prepare script
        model_dir = self.model_work_dir / "model"
        s = "#!/bin/bash\n"
        s += "set -e\n"
        s += f"source {Environment.conda_init_path}\n"
        s += f"conda activate {self.ENV_NAME}\n"
        s += "set -x\n"
        s += f"cd {self.SRC_DIR}\n"
        # Reducing test_batch_size to 4, otherwise it will delete some test data due to some bug
        s += f"""MKL_THREADING_LAYER=GNU CUDA_VISIBLE_DEVICES={gpu_id} python -W ignore '{self.SRC_DIR}/main/test.py' \\
               --data_workers 5 \\
               --dataset_name java \\
               --data_dir '{data_dir}/' \\
               --model_dir '{model_dir}' \\
               --model_name model \\
               --dev_src test/code.original_subtoken \\
               --dev_tgt test/javadoc.original \\
               --uncase True \\
               --max_examples -1 \\
               --max_src_len {self.config.max_src_len} \\
               --max_tgt_len {self.config.max_tgt_len} \\
               --test_batch_size 4 \\
               --beam_size 4 \\
               --n_best 1 \\
               --block_ngram_repeat 3 \\
               --stepwise_penalty False \\
               --coverage_penalty none \\
               --length_penalty none \\
               --beta 0 \\
               --gamma 0 \\
               --replace_unk
            """

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
        beam_res = IOUtils.load(model_dir / "model_beam.json", IOUtils.Format.jsonList)
        predictions = [[] for _ in range(len(dataset))]
        for x in beam_res:
            predictions[int(x["references"][0])] = x["predictions"][0].split(" ")

        # Delete temp files
        IOUtils.rm_dir(data_dir)
        IOUtils.rm(script_path)
        IOUtils.rm(stdout_path)
        IOUtils.rm(stderr_path)

        return predictions

    def save(self) -> None:
        # Save config and training status
        IOUtils.dump(self.model_work_dir / "config.json", IOUtils.jsonfy(self.config), IOUtils.Format.jsonPretty)
        IOUtils.dump(self.model_work_dir / "train_finished.json", self.train_finished)

        # Model should already be saved/checkpointed to the correct path
        return

    @classmethod
    def load(cls, model_work_dir) -> "CGModelBase":
        obj = TransformerACL20(model_work_dir)
        obj.config = IOUtils.dejsonfy(IOUtils.load(model_work_dir / "config.json"), TransformerACL20Config)
        obj.train_finished = IOUtils.load(model_work_dir / "train_finished.json")
        return obj
