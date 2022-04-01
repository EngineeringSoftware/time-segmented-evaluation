import inspect
from inspect import Parameter
from pathlib import Path
from typing import *

import copy
import numpy as np
import typing_inspect
from scipy import stats


class Utils:

    @classmethod
    def get_option_as_boolean(cls, options, opt, default=False, pop=False) -> bool:
        if opt not in options:
            return default
        else:
            # Due to limitations of CliUtils...
            value = options.get(opt, "false")
            if pop:
                del options[opt]
            return str(value).lower() != "false"

    @classmethod
    def get_option_as_list(cls, options, opt, default=None, pop=False) -> list:
        if opt not in options:
            return copy.deepcopy(default)
        else:
            lst = options[opt]
            if pop:
                del options[opt]
            if not isinstance(lst, list):
                lst = [lst]
            return lst

    @classmethod
    def get_option_and_pop(cls, options, opt, default=None) -> any:
        if opt in options:
            return options.pop(opt)
        else:
            return copy.deepcopy(default)

    # Summaries
    SUMMARIES_FUNCS: Dict[str, Callable[[Union[list, np.ndarray]], Union[int, float]]] = {
        "AVG": lambda l: np.mean(l) if len(l) > 0 else np.NaN,
        "SUM": lambda l: sum(l) if len(l) > 0 else np.NaN,
        "MAX": lambda l: max(l) if len(l) > 0 else np.NaN,
        "MIN": lambda l: min(l) if len(l) > 0 else np.NaN,
        "MEDIAN": lambda l: np.median(l) if len(l) > 0 and np.NaN not in l else np.NaN,
        "STDEV": lambda l: np.std(l) if len(l) > 0 else np.NaN,
        "MODE": lambda l: stats.mode(l).mode[0].item() if len(l) > 0 else np.NaN,
        "CNT": lambda l: len(l),
    }

    SUMMARIES_PRESERVE_INT: Dict[str, bool] = {
        "AVG": False,
        "SUM": True,
        "MAX": True,
        "MIN": True,
        "MEDIAN": False,
        "STDEV": False,
        "MODE": True,
        "CNT": True,
    }

    @classmethod
    def parse_cmd_options_for_type(
            cls,
            options: dict,
            typ: type,
            excluding_params: List[str] = None,
    ) -> Tuple[dict, dict, list]:
        """
        Parses the commandline options (got from seutil.CliUtils) based on the parameters
        and their types specified in typ.__init__.

        :param options: the commandline options got from seutil.CliUtils.
        :param typ: the type to initialize.
        :param excluding_params: the list of parameters that are not expected to be
            passed from commandline, by default ["self"].
        :return: two dictionaries and a list:
             a dictionary with options that can be sent to typ.__init__;
             a dictionary that contains the remaining options;
             a list of any missing options required by the typ.__init__.
        """
        if excluding_params is None:
            excluding_params = ["self"]

        accepted_options = {}
        unk_options = options
        missing_options = []

        signature = inspect.signature(typ.__init__)
        for param in signature.parameters.values():
            if param.name in excluding_params:
                continue

            if param.kind == Parameter.POSITIONAL_ONLY \
                    or param.kind == Parameter.VAR_KEYWORD \
                    or param.kind == Parameter.VAR_POSITIONAL:
                raise AssertionError(f"Class {typ.__name__} should not have '/', '**' '*'"
                                     f" parameters in order to be configured from commandline")

            if param.name not in unk_options:
                if param.default == Parameter.empty:
                    missing_options.append(param.name)
                    continue
                else:
                    # No need to insert anything to model_options
                    continue

            if param.annotation == bool:
                accepted_options[param.name] = Utils.get_option_as_boolean(unk_options, param.name, pop=True)
            elif typing_inspect.get_origin(param.annotation) == list:
                accepted_options[param.name] = Utils.get_option_as_list(unk_options, param.name, pop=True)
            elif typing_inspect.get_origin(param.annotation) == set:
                accepted_options[param.name] = set(Utils.get_option_as_list(unk_options, param.name, pop=True))
            elif typing_inspect.get_origin(param.annotation) == tuple:
                accepted_options[param.name] = tuple(Utils.get_option_as_list(unk_options, param.name, pop=True))
            else:
                accepted_options[param.name] = unk_options.pop(param.name)

        return accepted_options, unk_options, missing_options

    @classmethod
    def expect_dir_or_suggest_dvc_pull(cls, path: Path):
        if not path.is_dir():
            dvc_file = path.parent / (path.name+".dvc")
            if dvc_file.exists():
                print(f"{path} does not exist, but {dvc_file} exists. You probably want to dvc pull that file first?")
                print(f"# DVC command to run:")
                print(f"  dvc pull {dvc_file}")
                raise AssertionError(f"{path} does not exist but can be pulled from dvc.\n  dvc pull {dvc_file}")
            else:
                raise AssertionError(f"{path} does not exist.")

    @classmethod
    def suggest_dvc_add(cls, *paths: Path) -> str:
        s = f"# DVC commands:\n"
        s += f"  dvc add "
        for path in paths:
            s += str(path)
        s += "\n"
        return s
