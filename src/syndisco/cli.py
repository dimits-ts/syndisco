# SynDisco: Automated experiment creation and execution using only LLM agents
# Copyright (C) 2026 Dimitris Tsirmpas

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
"""
Command-line interface for SynDisco.
"""
import argparse
import sys
import typing

from . import __version__

VERSION_STRING = """
Syndisco {version}
Copyright (C) 2026 Dimitris Tsirmpas
License GPLv3+: GNU GPL version 3 or later <https://gnu.org/licenses/gpl.html>
This is free software: you are free to change and redistribute it.
There is NO WARRANTY, to the extent permitted by law.
"""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="syndisco",
        description="SynDisco command-line tools.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=VERSION_STRING.format(version=__version__),
    )
    commands = parser.add_subparsers(dest="command", metavar="command")

    view = commands.add_parser(
        "view",
        help="open SynDisco Viewer in your browser",
        description=(
            "Start a local web server for SynDisco Viewer and open it in "
            "your browser. If PATH is given (a folder of discussion JSON "
            "files, a single JSON file, or a .zip), it is loaded "
            "automatically. Files are served read-only."
        ),
    )
    view.add_argument(
        "path",
        nargs="?",
        default=None,
        help="folder, JSON file or .zip with discussion logs to load",
    )
    view.add_argument(
        "--host",
        default="127.0.0.1",
        help="address to bind to (default: 127.0.0.1, this computer only)",
    )
    view.add_argument(
        "--port",
        type=int,
        default=8765,
        help="port to use; the next free port is used if it is taken "
        "(default: 8765)",
    )
    view.add_argument(
        "--no-browser",
        action="store_true",
        help="do not open a browser window",
    )
    view.add_argument(
        "--show-prompts",
        action="store_true",
        help="show system prompts by default",
    )
    view.add_argument(
        "--verbose",
        action="store_true",
        help="log every request to the terminal",
    )
    return parser


def main(argv: typing.Sequence[str] | None = None) -> int:
    """
    Entry point of the ``syndisco`` command.

    :param argv: command-line arguments, defaults to ``sys.argv[1:]``
    :type argv: typing.Sequence[str] | None
    :return: the process exit code
    :rtype: int
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "view":
        from .viewer import server

        try:
            return server.run(
                path=args.path,
                host=args.host,
                port=args.port,
                open_browser=not args.no_browser,
                show_prompts=args.show_prompts,
                verbose=args.verbose,
            )
        except (FileNotFoundError, ValueError, OSError) as e:
            print(f"syndisco view: {e}", file=sys.stderr)
            return 2

    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
