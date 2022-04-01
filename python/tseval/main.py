import random
import sys
import time
from pathlib import Path

import pkg_resources
from seutil import CliUtils, IOUtils, LoggingUtils

from tseval.Environment import Environment
from tseval.Macros import Macros
from tseval.Utils import Utils

# Check seutil version
EXPECTED_SEUTIL_VERSION = "0.5.6"
if pkg_resources.get_distribution("seutil").version < EXPECTED_SEUTIL_VERSION:
    print(f"seutil version does not meet expectation! Expected version: {EXPECTED_SEUTIL_VERSION}, current installed version: {pkg_resources.get_distribution('seutil').version}", file=sys.stderr)
    print(f"Hint: either upgrade seutil, or modify the expected version (after confirmation that the version will work)", file=sys.stderr)
    sys.exit(-1)


logging_file = Macros.python_dir / "experiment.log"
LoggingUtils.setup(filename=str(logging_file))

logger = LoggingUtils.get_logger(__name__)


# ==========
# Data collection, sample

def collect_repos(**options):
    from tseval.collector.DataCollector import DataCollector
    DataCollector().search_github_java_repos()


def filter_repos(**options):
    from tseval.collector.DataCollector import DataCollector
    DataCollector().filter_repos(
        year_end=options.get("year_end", 2021),
        year_cnt=options.get("year_cnt", 3),
        loc_min=options.get("loc_min", 1e6),
        loc_max=options.get("loc_max", 2e6),
        star_min=options.get("star_min", 20),
    )


def collect_raw_data(**options):
    from tseval.collector.DataCollector import DataCollector
    DataCollector().collect_raw_data_projects(
        year_end=options.get("year_end", 2021),
        year_cnt=options.get("year_cnt", 3),
        skip_collected=Utils.get_option_as_boolean(options, "skip_collected"),
        project_names=Utils.get_option_as_list(options, "projects"),
    )


def process_raw_data(**options):
    from tseval.collector.DataCollector import DataCollector
    DataCollector().process_raw_data(
        year_end=options.get("year_end", 2021),
        year_cnt=options.get("year_cnt", 3),
    )


def get_splits(**options):
    from tseval.eval.EvalHelper import EvalHelper
    EvalHelper().get_splits(
        split_name=options["split"],
        seed=options.get("seed", 7),
        prj_val_ratio=options.get("prj_val_ratio", 0.1),
        prj_test_ratio=options.get("prj_test_ratio", 0.2),
        inprj_val_ratio=options.get("inprj_val_ratio", 0.1),
        inprj_test_ratio=options.get("inprj_test_ratio", 0.2),
        train_year=options.get("train_year", 2019),
        val_year=options.get("val_year", 2020),
        test_year=options.get("test_year", 2021),
        debug=Utils.get_option_as_boolean(options, "debug"),
    )


# ==========
# Machine learning

def prepare_envs(**options):
    which = Utils.get_option_as_list(options, "which")
    if which is None or "TransformerACL20" in which:
        from tseval.comgen.model.TransformerACL20 import TransformerACL20
        TransformerACL20.prepare_env()
    if which is None or "DeepComHybridESE19" in which:
        from tseval.comgen.model.DeepComHybridESE19 import DeepComHybridESE19
        DeepComHybridESE19.prepare_env()
    if which is None or "Code2SeqICLR19" in which:
        from tseval.metnam.model.Code2SeqICLR19 import Code2SeqICLR19
        Code2SeqICLR19.prepare_env()
    if which is None or "Code2VecPOPL19" in which:
        from tseval.metnam.model.Code2VecPOPL19 import Code2VecPOPL19
        Code2VecPOPL19.prepare_env()


def exp_prepare(**options):
    from tseval.eval.EvalHelper import EvalHelper
    EvalHelper().exp_prepare(**options)


def exp_train(**options):
    from tseval.eval.EvalHelper import EvalHelper
    EvalHelper().exp_train(**options)


def exp_eval(**options):
    from tseval.eval.EvalHelper import EvalHelper
    EvalHelper().exp_eval(**options)


def exp_compute_metrics(**options):
    from tseval.eval.EvalHelper import EvalHelper
    EvalHelper().exp_compute_metrics(**options)


# ==========
# Table & Plot

def make_tables(**options):
    from tseval.Table import Table
    Table().make_tables(options)


def make_plots(**options):
    from tseval.Plot import Plot
    Plot().make_plots(options)


# ==========
# Metrics collection

def collect_metrics(**options):
    from tseval.collector.MetricsCollector import MetricsCollector

    mc = MetricsCollector()
    mc.collect_metrics(**options)
    return


# ==========
# Collect and analyze results

def analyze_check_files(**options):
    from tseval.collector.ExperimentsAnalyzer import ExperimentsAnalyzer
    ExperimentsAnalyzer(
        exps_spec_path=Path(options["exps"]),
        output_prefix=options.get("output"),
    ).check_files()


def analyze_recompute_metrics(**options):
    from tseval.collector.ExperimentsAnalyzer import ExperimentsAnalyzer
    ExperimentsAnalyzer(
        exps_spec_path=Path(options["exps"]),
        output_prefix=options.get("output"),
    ).recompute_metrics()


def analyze_extract_metrics(**options):
    from tseval.collector.ExperimentsAnalyzer import ExperimentsAnalyzer
    ExperimentsAnalyzer(
        exps_spec_path=Path(options["exps"]),
        output_prefix=options.get("output"),
    ).extract_metrics()


def analyze_sign_test(**options):
    from tseval.collector.ExperimentsAnalyzer import ExperimentsAnalyzer
    ExperimentsAnalyzer(
        exps_spec_path=Path(options["exps"]),
        output_prefix=options.get("output"),
    ).sign_test_default()


def analyze_make_tables(**options):
    from tseval.collector.ExperimentsAnalyzer import ExperimentsAnalyzer
    ExperimentsAnalyzer(
        exps_spec_path=Path(options["exps"]),
        output_prefix=options.get("output"),
    ).make_tables_default()


def analyze_make_plots(**options):
    from tseval.collector.ExperimentsAnalyzer import ExperimentsAnalyzer
    ExperimentsAnalyzer(
        exps_spec_path=Path(options["exps"]),
        output_prefix=options.get("output"),
    ).make_plots_default()


def analyze_sample_results(**options):
    from tseval.collector.ExperimentsAnalyzer import ExperimentsAnalyzer
    ExperimentsAnalyzer(
        exps_spec_path=Path(options["exps"]),
        output_prefix=options.get("output"),
    ).sample_results(
        seed=options.get("seed", 7),
        count=options.get("count", 100),
    )


def analyze_extract_data_similarities(**options):
    from tseval.collector.ExperimentsAnalyzer import ExperimentsAnalyzer
    ExperimentsAnalyzer(
        exps_spec_path=Path(options["exps"]),
        output_prefix=options.get("output"),
    ).extract_data_similarities()


def analyze_near_duplicates(**options):
    from tseval.collector.ExperimentsAnalyzer import ExperimentsAnalyzer
    ExperimentsAnalyzer(
        exps_spec_path=Path(options["exps"]),
        output_prefix=options.get("output"),
    ).filter_near_duplicates_and_analyze(
        code_sim_threshold=options["code_sim"],
        nl_sim_threshold=options["nl_sim"],
        config_name=options["config"],
        only_tables_plots=Utils.get_option_as_boolean(options, "only_tables_plots", default=False),
    )


# ==========
# Main

def normalize_options(opts: dict) -> dict:
    # Set a different log file
    if "log_path" in opts:
        logger.info(f"Switching to log file {opts['log_path']}")
        LoggingUtils.setup(filename=opts['log_path'])

    # Set debug mode
    if "debug" in opts and str(opts["debug"]).lower() != "false":
        Environment.is_debug = True
        logger.debug("Debug mode on")
        logger.debug(f"Command line options: {opts}")

    # Set parallel mode - all automatic installations are disabled
    if "parallel" in opts and str(opts["parallel"]).lower() != "false":
        Environment.is_parallel = True
        logger.warning(f"Parallel mode on")

    # Set/report random seed
    if "random_seed" in opts:
        Environment.random_seed = int(opts["random_seed"])
    else:
        Environment.random_seed = time.time_ns()

    random.seed(Environment.random_seed)
    logger.info(f"Random seed is {Environment.random_seed}")
    return opts


if __name__ == "__main__":
    CliUtils.main(sys.argv[1:], globals(), normalize_options)
