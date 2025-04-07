"""
SynDisco: Automated experiment creation and execution using only LLM agents
Copyright (C) 2025 Dimitris Tsirmpas

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

You may contact the author at tsirbasdim@gmail.com
"""

import time
import logging
import typing
import warnings
import functools
from pathlib import Path

import coloredlogs

from . import file_util


logger = logging.getLogger(Path(__file__).name)


def logging_setup(
    print_to_terminal: bool,
    write_to_file: bool,
    logs_dir: typing.Optional[str | Path] = None,
    level: str = "debug",
    use_colors: bool = True,
    log_warnings: bool = True,
) -> None:
    """
    Create the logger configuration.

    :param print_to_terminal: whether to print logs to the screen
    :type print_to_terminal: bool
    :param write_to_file: whether to write logs to a file.
        Needs logs_dir to be specified.
    :type write_to_file: bool
    :param logs_dir: the directory where the logs will be placed,
        defaults to None
    :type logs_dir: typing.Optional[str  |  Path], optional
    :param level: the logging level, defaults to logging.DEBUG
    :param use_colors: whether to color the output.
    :type use_colors: bool, defaults to True
    :param log_warnings: whether to log library warnings
    :type log_warnings: bool, defaults to True
    """
    if not print_to_terminal and logs_dir is None:
        warnings.warn(
            "Warning: Both screen-printing and file-printing has "
            "been disabled. No logs will be recorded for this session."
        )

    level = _str_to_log_level(level)  # type: ignore
    handlers = []
    if print_to_terminal:
        handlers.append(logging.StreamHandler())

    if write_to_file:
        if logs_dir is None:
            warnings.warn(
                "Warning: No logs directory provided ."
                "Disabling logging to file."
            )
        else:
            filename = file_util.generate_datetime_filename(
                logs_dir, file_ending=".log", timestamp_format="%y-%m-%d"
            )
            handlers.append(logging.FileHandler(filename))

    logging.basicConfig(
        handlers=handlers,
        level=level,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if use_colors:
        coloredlogs.install(level=level)

    logging.captureWarnings(log_warnings)


# https://stackoverflow.com/questions/1622943/timeit-versus-timing-decorator
def timing(f: typing.Callable) -> typing.Any:
    """
    Decorator which logs the execution time of a function.

    :param f: the function to be timed
    :type f: Function
    :return: the result of the function
    :rtype: _type_
    """

    @functools.wraps(f)
    def wrap(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        exec_time_mins = (te - ts) / 60
        logger.info(
            f"Procedure {f.__name__} executed in {exec_time_mins:2.4f} minutes"
        )
        return result

    return wrap


def _str_to_log_level(level_str: str):
    match level_str.lower().strip():
        case "debug":
            return logging.DEBUG
        case "info":
            return logging.INFO
        case "not_set":
            return logging.NOTSET
        case "warning":
            return logging.WARNING
        case "warn":
            return logging.WARNING
        case "error":
            return logging.ERROR
        case "critical":
            return logging.CRITICAL
        case _:
            logger.warning(
                f"Unrecognized log level {level_str}. Defaulting to NOT_SET"
            )
            return logging.NOTSET
