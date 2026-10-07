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
Local web server behind ``syndisco view``.

The server has two jobs:

1. Serve the static viewer bundled in this package.
2. Optionally expose one folder, JSON file or .zip of discussion logs,
   read-only, so the viewer loads it automatically.

It only uses the Python standard library. Data files are restricted to
``.json`` files inside the chosen folder (symlinks and ``..`` cannot escape
it), and requests whose ``Host`` header does not match the server are
refused, which protects local data from DNS-rebinding attacks by web pages.
"""
import errno
import functools
import http.server
import importlib.resources
import io
import ipaddress
import json
import shutil
import socket
import sys
import threading
import typing
import urllib.parse
import webbrowser
import zipfile
from pathlib import Path

from .. import __version__

#: Number of consecutive ports tried when the requested port is taken.
PORT_ATTEMPTS = 20

_JSON_TYPE = "application/json; charset=utf-8"


class DataSource:
    """
    A folder, JSON file or zip archive exposed read-only to the viewer.

    :param path: the folder, ``.json`` file or ``.zip`` file to expose.
    :type path: str | Path
    :raises FileNotFoundError: if *path* does not exist.
    :raises ValueError: if *path* is a file that is neither JSON nor zip.
    """

    def __init__(self, path: str | Path) -> None:
        given = Path(path).expanduser()
        if not given.exists():
            raise FileNotFoundError(f"{given} does not exist")
        self.path = given.resolve()
        if self.path.is_dir():
            self.kind = "dir"
        elif self.path.suffix.lower() == ".json":
            self.kind = "file"
        elif self.path.suffix.lower() == ".zip":
            self.kind = "zip"
        else:
            raise ValueError(
                f"{given} must be a folder, a .json file or a .zip file"
            )

    @property
    def label(self) -> str:
        """Human-readable name shown in the viewer."""
        return self.path.name or str(self.path)

    def dataset_url(self) -> str:
        """
        URL of the dataset, relative to the viewer page.

        Folders are offered as one zip so the browser makes a single
        request instead of one per file.
        """
        return "local/data.zip"

    def zip_bytes(self) -> bytes:
        """
        The exposed files as an uncompressed zip archive.

        :return: the archive; for a zip source, the file itself.
        :rtype: bytes
        """
        if self.kind == "zip":
            return self.path.read_bytes()
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as archive:
            for rel in self.list_files():
                target = self.resolve(rel)
                if target is not None:
                    archive.write(target, arcname=rel)
        return buffer.getvalue()

    def list_files(self) -> list[str]:
        """
        List the JSON files exposed by this source.

        :return: POSIX-style paths relative to the folder, sorted.
        :rtype: list[str]
        """
        if self.kind == "file":
            return [self.path.name]
        if self.kind == "zip":
            return []
        files = []
        for candidate in self.path.rglob("*"):
            rel = candidate.relative_to(self.path)
            if any(part.startswith(".") for part in rel.parts):
                continue
            if candidate.suffix.lower() != ".json":
                continue
            if self.resolve(rel.as_posix()) is not None:
                files.append(rel.as_posix())
        return sorted(files)

    def resolve(self, rel: str) -> Path | None:
        """
        Map a relative path from a request to a file on disk.

        :param rel: the requested path, relative to the source.
        :type rel: str
        :return: the file, or None if it is missing or not allowed.
        :rtype: Path | None
        """
        if not rel or "\x00" in rel or rel.startswith(("/", "\\")):
            return None
        if self.kind == "file":
            return self.path if rel == self.path.name else None
        if self.kind != "dir":
            return None
        try:
            candidate = (self.path / rel).resolve()
        except (OSError, RuntimeError):
            return None
        if not candidate.is_relative_to(self.path):
            return None
        inside = candidate.relative_to(self.path).parts
        if any(part.startswith(".") for part in inside):
            return None
        if candidate.suffix.lower() != ".json" or not candidate.is_file():
            return None
        return candidate

    def manifest(self) -> dict[str, typing.Any]:
        """
        The manifest the viewer reads to find every file.

        File URLs are relative to the manifest's own URL.
        """
        return {
            "name": self.label,
            "files": [
                {"path": rel, "url": "files/" + urllib.parse.quote(rel)}
                for rel in self.list_files()
            ],
        }


class ViewerRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Serves the static viewer plus the read-only ``/local/`` data routes."""

    server_version = "SynDiscoViewer"

    # set by make_handler()
    source: DataSource | None = None
    config_overrides: dict[str, typing.Any] = {}
    allowed_hosts: set[str] | None = None
    verbose: bool = False

    def end_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        super().end_headers()

    def log_message(self, format: str, *args: typing.Any) -> None:
        if self.verbose:
            super().log_message(format, *args)

    def list_directory(self, path):  # type: ignore[override]
        self.send_error(404, "Not found")
        return None

    def do_GET(self) -> None:
        self._dispatch(head_only=False)

    def do_HEAD(self) -> None:
        self._dispatch(head_only=True)

    # -- routing ---------------------------------------------------------

    def _dispatch(self, head_only: bool) -> None:
        if not self._host_allowed():
            self.send_error(403, "Host not allowed")
            return

        route = urllib.parse.unquote(urllib.parse.urlsplit(self.path).path)

        if route == "/config.json":
            self._send_json(self._config(), head_only)
        elif route == "/datasets/index.json" and self.source is not None:
            self._send_json(self._datasets_index(), head_only)
        elif route.startswith("/local/"):
            self._serve_local(route[len("/local/"):], head_only)
        elif head_only:
            super().do_HEAD()
        else:
            super().do_GET()

    def _host_allowed(self) -> bool:
        if self.allowed_hosts is None:
            return True
        host = (self.headers.get("Host") or "").strip().lower()
        return host in self.allowed_hosts

    def _config(self) -> dict[str, typing.Any]:
        config: dict[str, typing.Any] = {}
        config_path = Path(self.directory) / "config.json"
        try:
            config = json.loads(config_path.read_text(encoding="utf8"))
        except (OSError, ValueError):
            pass
        config.update(self.config_overrides)
        return config

    def _datasets_index(self) -> dict[str, typing.Any]:
        assert self.source is not None
        return {
            "datasets": [
                {
                    "id": "local",
                    "name": self.source.label,
                    "description": "Served from this computer by "
                    "syndisco view.",
                    "url": self.source.dataset_url(),
                }
            ],
            "autoload": "local",
        }

    def _serve_local(self, rel: str, head_only: bool) -> None:
        source = self.source
        if source is None:
            self.send_error(404, "No data folder is being served")
            return

        if rel == "manifest.json" and source.kind != "zip":
            self._send_json(source.manifest(), head_only)
            return

        if rel == "data.zip":
            if source.kind == "zip":
                self._send_file(source.path, "application/zip", head_only)
            else:
                self._send_bytes(source.zip_bytes(), "application/zip", head_only)
            return

        if rel.startswith("files/"):
            target = source.resolve(rel[len("files/"):])
            if target is not None:
                self._send_file(target, _JSON_TYPE, head_only)
                return

        self.send_error(404, "Not found")

    # -- responses -------------------------------------------------------

    def _send_json(self, payload: typing.Any, head_only: bool) -> None:
        self._send_bytes(json.dumps(payload).encode("utf8"), _JSON_TYPE, head_only)

    def _send_bytes(self, body: bytes, ctype: str, head_only: bool) -> None:
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if not head_only:
            self.wfile.write(body)

    def _send_file(self, path: Path, ctype: str, head_only: bool) -> None:
        try:
            handle = open(path, "rb")
        except OSError:
            self.send_error(404, "Not found")
            return
        with handle:
            size = path.stat().st_size
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(size))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            if not head_only:
                shutil.copyfileobj(handle, self.wfile)


def static_dir() -> Path:
    """
    Location of the bundled viewer files.

    :return: the directory containing the viewer's ``index.html``.
    :rtype: Path
    """
    return Path(str(importlib.resources.files(__package__) / "static"))


def _is_loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _is_wildcard(host: str) -> bool:
    return host in ("", "0.0.0.0", "::")


def _allowed_hosts(host: str, port: int) -> set[str] | None:
    """Host headers accepted by the server; None accepts any."""
    if not _is_loopback(host):
        return None
    names = {"localhost", "127.0.0.1", "[::1]", host.lower()}
    if ":" in host and not host.startswith("["):
        names.add(f"[{host.lower()}]")
    return {f"{name}:{port}" for name in names} | names


def make_handler(
    directory: Path,
    source: DataSource | None,
    config_overrides: dict[str, typing.Any],
    allowed_hosts: set[str] | None,
    verbose: bool,
) -> typing.Callable[..., ViewerRequestHandler]:
    """Create a request handler bound to the given settings."""
    handler_cls = type(
        "BoundViewerRequestHandler",
        (ViewerRequestHandler,),
        {
            "source": source,
            "config_overrides": dict(config_overrides),
            "allowed_hosts": allowed_hosts,
            "verbose": verbose,
        },
    )
    return functools.partial(handler_cls, directory=str(directory))


def create_server(
    path: str | Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8765,
    show_prompts: bool = False,
    verbose: bool = False,
) -> http.server.ThreadingHTTPServer:
    """
    Create (but do not start) the viewer server.

    :param path: folder, JSON file or zip to expose, defaults to None
    :type path: str | Path | None
    :param host: address to bind to, defaults to "127.0.0.1"
    :type host: str
    :param port: preferred port; 0 picks any free port. If the port is
        taken, the next ones are tried. Defaults to 8765.
    :type port: int
    :param show_prompts: show system prompts by default, defaults to False
    :type show_prompts: bool
    :param verbose: log every request, defaults to False
    :type verbose: bool
    :raises FileNotFoundError: if *path* or the viewer files are missing.
    :raises OSError: if no port could be bound.
    :return: the server; call ``serve_forever()`` to start it.
    :rtype: http.server.ThreadingHTTPServer
    """
    directory = static_dir()
    if not (directory / "index.html").is_file():
        raise FileNotFoundError(
            f"viewer files not found in {directory}; the installation may "
            "be incomplete"
        )
    source = DataSource(path) if path is not None else None
    overrides: dict[str, typing.Any] = {"version": __version__}
    if show_prompts:
        overrides["showPrompts"] = True

    server_cls = http.server.ThreadingHTTPServer
    if ":" in host:
        server_cls = type(
            "IPv6Server", (server_cls,), {"address_family": socket.AF_INET6}
        )

    ports = [port] if port == 0 else range(port, port + PORT_ATTEMPTS)
    last_error: OSError | None = None
    for candidate in ports:
        try:
            # the handler needs the final port for Host checks, so bind
            # first and attach the handler afterwards
            server = server_cls((host, candidate), ViewerRequestHandler)
        except OSError as e:
            if e.errno != errno.EADDRINUSE:
                raise
            last_error = e
            continue
        bound_port = server.server_address[1]
        server.RequestHandlerClass = make_handler(  # type: ignore[assignment]
            directory=directory,
            source=source,
            config_overrides=overrides,
            allowed_hosts=_allowed_hosts(host, bound_port),
            verbose=verbose,
        )
        return server
    raise OSError(
        f"ports {port}-{port + PORT_ATTEMPTS - 1} are all in use; "
        "choose another with --port"
    ) from last_error


def server_url(server: http.server.ThreadingHTTPServer) -> str:
    """The address a browser should open for *server*."""
    host, port = server.server_address[:2]
    host = str(host)
    if _is_wildcard(host):
        host = "127.0.0.1"
    elif ":" in host:
        host = f"[{host}]"
    return f"http://{host}:{port}/"


def run(
    path: str | Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
    show_prompts: bool = False,
    verbose: bool = False,
) -> int:
    """
    Serve the viewer until interrupted with Ctrl+C.

    See :func:`create_server` for the parameters.

    :param open_browser: open the viewer in the default browser,
        defaults to True
    :type open_browser: bool
    :return: the process exit code
    :rtype: int
    """
    server = create_server(
        path=path,
        host=host,
        port=port,
        show_prompts=show_prompts,
        verbose=verbose,
    )
    url = server_url(server)

    if not _is_loopback(host):
        print(
            f"Warning: listening on {host}. Other devices on your network "
            "can open the viewer and read the served files.",
            file=sys.stderr,
        )
    print(f"SynDisco Viewer is running at {url}")
    if path is not None:
        print(f"Serving {Path(path).expanduser().resolve()} (read-only)")
    print("Press Ctrl+C to stop.")

    if open_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
    return 0
