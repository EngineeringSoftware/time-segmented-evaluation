from pathlib import Path

from seutil import BashUtils, IOUtils, LoggingUtils, MiscUtils

from tseval.Macros import Macros


class Environment:
    logger = LoggingUtils.get_logger(__name__, LoggingUtils.INFO)

    # ----------
    # Environment variables
    # ----------
    is_debug: bool = False
    random_seed: int = 1
    is_parallel: bool = False

    # ----------
    # Conda & CUDA
    # ----------
    conda_env = "tseval"

    @classmethod
    def get_conda_init_path(cls) -> str:
        which_conda = BashUtils.run("which conda").stdout.strip()
        if len(which_conda) == 0:
            raise RuntimeError(f"Cannot detect conda environment!")
        return str(Path(which_conda).parent.parent/"etc"/"profile.d"/"conda.sh")

    conda_init_path_cached = None

    @MiscUtils.classproperty
    def conda_init_path(cls):
        if cls.conda_init_path_cached is None:
            cls.conda_init_path_cached = cls.get_conda_init_path()
        return cls.conda_init_path_cached

    @classmethod
    def get_cuda_version(cls) -> str:
        which_nvidia_smi = BashUtils.run("which nvidia-smi").stdout.strip()
        if len(which_nvidia_smi) == 0:
            return "cpu"
        else:
            cuda_version_number = BashUtils.run(r'nvcc -V | grep "release" | sed -E "s/.*release ([^,]+),.*/\1/"').stdout.strip()
            if cuda_version_number.startswith("10.0"):
                return "cu100"
            elif cuda_version_number.startswith("10.1"):
                return "cu101"
            elif cuda_version_number.startswith("10.2"):
                return "cu102"
            elif cuda_version_number.startswith("11.0"):
                return "cu110"
            else:
                raise RuntimeError(f"Unsupported cuda version {cuda_version_number}!")

    cuda_version_cached = None

    @MiscUtils.classproperty
    def cuda_version(cls):
        if cls.cuda_version_cached is None:
            cls.cuda_version_cached = cls.get_cuda_version()
        return cls.cuda_version_cached

    @classmethod
    def get_cuda_toolkit_spec(cls):
        cuda_version = cls.cuda_version
        if cuda_version == "cpu":
            return "cpuonly"
        elif cuda_version == "cu100":
            return "cudatoolkit=10.1"
        elif cuda_version == "cu101":
            return "cudatoolkit=10.1"
        elif cuda_version == "cu102":
            return "cudatoolkit=10.2"
        elif cuda_version == "cu110":
            return "cudatoolkit=11.0"
        else:
            raise RuntimeError(f"Unexpected cuda version {cuda_version}!")

    # ----------
    # Tools
    # ----------

    collector_installed = False
    collector_jar = str(Macros.collector_dir / "target" / f"collector-{Macros.collector_version}.jar")

    @classmethod
    def require_collector(cls):
        if cls.is_parallel:
            return
        if not cls.collector_installed:
            cls.logger.info("Require collector, installing ...")
            with IOUtils.cd(Macros.collector_dir):
                BashUtils.run(f"mvn clean install -DskipTests", expected_return_code=0)
            cls.collector_installed = True
        else:
            cls.logger.debug("Require collector, and already installed")
        return
