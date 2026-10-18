"""A disk cache for GitHub API responses, revalidated with ETags.

Every cached response is stored with GitHub's ``ETag``. The next request for the
same URL sends it as ``If-None-Match``; GitHub answers ``304 Not Modified``
when nothing changed, and the stored body is served. So a repeated question
never works from stale data, and an unchanged page costs a round trip with no
body. GitHub documents that a ``304`` made with a token does not count against
the rate limit.

Entries live under ``<cache root>/v<package version>-s<schema>/``, one JSON
file per URL named by the SHA-256 of the URL. A new release or a change to the
stored shape starts a fresh namespace, and older namespaces are removed on
first use, so an upgrade never reads entries an older version wrote.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import httpx

#: Bump when the stored entry's shape changes.
CACHE_SCHEMA = 1

#: Response headers kept with an entry: what pagination and rate limiting read.
KEPT_HEADERS = ("content-type", "etag", "link")


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("graphweave")
    except Exception:  # noqa: BLE001 - not installed as a distribution
        return "dev"


def namespace_dir(root: str | Path) -> Path:
    """The versioned folder entries go in, with stale siblings removed."""
    root = Path(root)
    current = root / f"v{_version()}-s{CACHE_SCHEMA}"
    if root.is_dir():
        for sibling in root.iterdir():
            if sibling.is_dir() and sibling.name.startswith("v") and sibling != current:
                shutil.rmtree(sibling, ignore_errors=True)
    current.mkdir(parents=True, exist_ok=True)
    return current


class CachingTransport(httpx.BaseTransport):
    """Wraps a transport: GETs are revalidated against a disk cache by ETag."""

    def __init__(self, directory: str | Path, inner: httpx.BaseTransport | None = None) -> None:
        self.directory = namespace_dir(directory)
        self.inner = inner or httpx.HTTPTransport()
        #: Counters for a run: served from cache after a 304, and full fetches.
        self.revalidated = 0
        self.fetched = 0

    def _path(self, url: str) -> Path:
        return self.directory / f"{hashlib.sha256(url.encode('utf-8')).hexdigest()}.json"

    def _read(self, path: Path) -> dict | None:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def _write(self, path: Path, entry: dict) -> None:
        # Written whole then moved into place, so a crash never leaves half an entry.
        handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                json.dump(entry, stream)
            os.replace(temporary, path)
        except OSError:
            Path(temporary).unlink(missing_ok=True)

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        if request.method != "GET":
            return self.inner.handle_request(request)

        url = str(request.url)
        path = self._path(url)
        entry = self._read(path)
        if entry and entry.get("etag"):
            request.headers["If-None-Match"] = entry["etag"]

        response = self.inner.handle_request(request)
        if response.status_code == 304 and entry:
            response.close()
            self.revalidated += 1
            return httpx.Response(
                200,
                headers=entry["headers"],
                content=entry["body"].encode("utf-8"),
                request=request,
            )

        if response.status_code == 200 and response.headers.get("etag"):
            body = response.read()
            self._write(path, {
                "url": url,
                "etag": response.headers["etag"],
                "headers": {key: response.headers[key] for key in KEPT_HEADERS if key in response.headers},
                "body": body.decode("utf-8"),
            })
            self.fetched += 1
            # The body is already decoded, so encoding and length headers would
            # make it be decoded twice.
            headers = [(key, value) for key, value in response.headers.items()
                       if key.lower() not in {"content-encoding", "content-length", "transfer-encoding"}]
            return httpx.Response(200, headers=headers, content=body, request=request)
        return response
