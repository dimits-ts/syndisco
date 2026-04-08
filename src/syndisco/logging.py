"""
Module handling logging for LLM discussion and annotation tasks.
"""

# SynDisco: Automated experiment creation and execution using only LLM agents
# Copyright (C) 2025 Dimitris Tsirmpas

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# You may contact the author at dim.tsirmpas@aueb.gr

import time
import logging as pylog
import typing
import warnings
import functools
from pathlib import Path
from logging.handlers import TimedRotatingFileHandler

import coloredlogs


logger = pylog.getLogger(Path(__file__).name)


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
        handlers.append(pylog.StreamHandler())

    if write_to_file:
        if logs_dir is None:
            warnings.warn(
                "Warning: No logs directory provided ."
                "Disabling logging to file."
            )
        else:
            handlers.append(_get_file_handler(Path(logs_dir)))

    pylog.basicConfig(
        handlers=handlers,
        level=level,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if use_colors:
        coloredlogs.install(level=level)

    pylog.captureWarnings(log_warnings)


def _str_to_log_level(level_str: str):
    match level_str.lower().strip():
        case "debug":
            return pylog.DEBUG
        case "info":
            return pylog.INFO
        case "not_set":
            return pylog.NOTSET
        case "warning":
            return pylog.WARNING
        case "warn":
            return pylog.WARNING
        case "error":
            return pylog.ERROR
        case "critical":
            return pylog.CRITICAL
        case _:
            logger.warning(
                f"Unrecognized log level {level_str}. Defaulting to NOT_SET"
            )
            return pylog.NOTSET


def _get_file_handler(logs_dir: Path):
    logfile_path = logs_dir / "log"  # base filename, extension gets added
    file_handler = TimedRotatingFileHandler(
        filename=logfile_path,
        when="midnight",
        interval=1,
        backupCount=7,
        encoding="utf-8",
        utc=True,
    )
    file_handler.suffix = "%y-%m-%d.log"
    return file_handler
