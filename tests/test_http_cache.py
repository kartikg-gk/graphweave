from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import httpx

from graphrag.ingestion.http_cache import CachingTransport, namespace_dir

PAGE = [{"number": 1, "title": "first"}]


class FakeGitHub:
    """Answers 304 when the client already holds the current ETag."""

    def __init__(self) -> None:
        self.etag = '"v1"'
        self.seen: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.seen.append(request)
        if request.headers.get("If-None-Match") == self.etag:
            return httpx.Response(304, headers={"etag": self.etag})
        body = json.dumps(PAGE if self.etag == '"v1"' else [{"number": 2, "title": "second"}])
        return httpx.Response(200, headers={"etag": self.etag, "link": '<https://x/next>; rel="next"'},
                              content=body.encode())


def client(folder: str, github: FakeGitHub) -> tuple[httpx.Client, CachingTransport]:
    transport = CachingTransport(folder, inner=httpx.MockTransport(github))
    return httpx.Client(base_url="https://api.github.com", transport=transport), transport


class CacheTests(unittest.TestCase):
    def test_an_unchanged_page_is_revalidated_and_served_from_disk(self):
        with TemporaryDirectory() as folder:
            github = FakeGitHub()
            session, transport = client(folder, github)
            first = session.get("/repos/o/r/pulls")
            second = session.get("/repos/o/r/pulls")
            self.assertEqual(first.json(), PAGE)
            self.assertEqual(second.status_code, 200)
            self.assertEqual(second.json(), PAGE)
            self.assertEqual(github.seen[1].headers["If-None-Match"], '"v1"')
            self.assertEqual((transport.fetched, transport.revalidated), (1, 1))
            # Pagination still works from a cached page.
            self.assertEqual(second.links["next"]["url"], "https://x/next")

    def test_a_changed_page_replaces_the_entry(self):
        with TemporaryDirectory() as folder:
            github = FakeGitHub()
            session, _ = client(folder, github)
            session.get("/repos/o/r/pulls")
            github.etag = '"v2"'
            self.assertEqual(session.get("/repos/o/r/pulls").json()[0]["number"], 2)
            self.assertEqual(session.get("/repos/o/r/pulls").json()[0]["number"], 2)

    def test_the_cache_survives_a_new_session(self):
        with TemporaryDirectory() as folder:
            github = FakeGitHub()
            client(folder, github)[0].get("/repos/o/r/pulls")
            session, transport = client(folder, github)
            self.assertEqual(session.get("/repos/o/r/pulls").json(), PAGE)
            self.assertEqual(transport.revalidated, 1)

    def test_other_versions_are_cleared(self):
        with TemporaryDirectory() as folder:
            stale = Path(folder) / "v0.0.1-s1"
            stale.mkdir()
            (stale / "old.json").write_text("{}")
            current = namespace_dir(folder)
            self.assertFalse(stale.exists())
            self.assertTrue(current.is_dir())

    def test_errors_are_not_cached(self):
        with TemporaryDirectory() as folder:
            calls = []

            def failing(request: httpx.Request) -> httpx.Response:
                calls.append(request)
                return httpx.Response(403, json={"message": "rate limited"})

            transport = CachingTransport(folder, inner=httpx.MockTransport(failing))
            session = httpx.Client(base_url="https://api.github.com", transport=transport)
            self.assertEqual(session.get("/x").status_code, 403)
            self.assertEqual(session.get("/x").status_code, 403)
            self.assertNotIn("If-None-Match", calls[1].headers)


if __name__ == "__main__":
    unittest.main()
