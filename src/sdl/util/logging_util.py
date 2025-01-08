import logging
import typing
import warnings
from pathlib import Path

import file_util


def log_setup(
    print_to_screen: bool,
    write_to_file: bool,
    logs_dir: typing.Optional[str | Path] = None,
    level=logging.DEBUG,
) -> None:
    """
    Create the logger configuration.

    :param print_to_screen: whether to print logs to the screen
    :type print_to_screen: bool
    :param write_to_file: whether to write logs to a file. Needs logs_dir to be specified.
    :type write_to_file: bool
    :param logs_dir: the directory where the logs will be placed, defaults to None
    :type logs_dir: typing.Optional[str  |  Path], optional
    :param level: the logging level, defaults to logging.DEBUG
    """
    if not print_to_screen and logs_dir is None:
        warnings.warn(
            "Warning: Both screen-printing and file-printing has been disabled. "
            "No logs will be recorded for this session."
        )

    handlers = []
    if print_to_screen:
        handlers.append(logging.StreamHandler())

    if write_to_file:
        if logs_dir is None:
            warnings.warn(
                "Warning: No logs directory provided. Disabling logging to file."
            )
        else:
            filename = file_util.generate_datetime_filename(
                logs_dir, file_ending=".log"
            )
            handlers.append(logging.FileHandler(filename))

    logging.basicConfig(handlers=handlers, level=level)
