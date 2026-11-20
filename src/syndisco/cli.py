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
Command-line interface for SynDisco, currently only used for the viewer.
"""
import argparse
import sys
import typing
import webbrowser
import zipfile

from . import __version__

VERSION_STRING = """
Syndisco {version}
Copyright (C) 2026 Dimitris Tsirmpas
License GPLv3+: GNU GPL version 3 or later <https://gnu.org/licenses/gpl.html>
This is free software: you are free to change and redistribute it.
There is NO WARRANTY, to the extent permitted by law.
"""


SOURCE_HELP = "folder, JSON file or .zip with discussion logs"


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
        help="open discussion logs in SynDisco Viewer",
        description=(
            "Write a SynDisco Viewer page with the discussions in SOURCE to "
            "a temporary file and open it in your browser. Use 'syndisco "
            "export' to keep the page."
        ),
    )
    view.add_argument("source", help=SOURCE_HELP)

    export_cmd = commands.add_parser(
        "export",
        help="save discussion logs as a SynDisco Viewer page",
        description=(
            "Write a self-contained SynDisco Viewer page with the "
            "discussions in SOURCE to TARGET. The page works offline and "
            "can be shared as a single file."
        ),
    )
    export_cmd.add_argument("source", help=SOURCE_HELP)
    export_cmd.add_argument(
        "target", help="HTML file to write; an existing file is overwritten"
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

    if args.command in ("view", "export"):
        from .viewer import export

        try:
            if args.command == "view":
                page = export.export_temporary(args.source)
            else:
                page = export.export(args.source, args.target)
        except (OSError, ValueError, zipfile.BadZipFile) as e:
            print(f"syndisco {args.command}: {e}", file=sys.stderr)
            return 2

        print(f"Wrote {page}")
        if args.command == "view":
            webbrowser.open(page.resolve().as_uri())
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
