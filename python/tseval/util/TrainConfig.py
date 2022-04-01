from typing import get_type_hints

from recordclass import RecordClass


class TrainConfig(RecordClass):
    train_session_time: int = 20 * 3600
    gpu_id: int = 0

    @classmethod
    def get_train_config_from_cmd_options(cls, options: dict) -> "TrainConfig":
        """
        Gets a TrainConfig from the command line options (the options will be modified
        in place to remove the parsed fields).
        """
        field_values = {}
        for f, t in get_type_hints(cls).items():
            if f in options:
                field_values[f] = t(options.pop(f))
        return cls(**field_values)
