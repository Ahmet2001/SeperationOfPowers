from __future__ import annotations

import json
import os
import unittest
from types import ModuleType
from unittest.mock import MagicMock, patch

import cli
from runtime.web_tools import (
    MAX_PAGE_CHARS,
    MAX_SEARCH_RESULTS,
    SEARXNG_URL_ENV,
    WebExecutor,
    searxng_url,
)


class FakeWebError(Exception):
    pass


def fake_web_module() -> ModuleType:
    module = ModuleType("ethosoftlib.web")
    module.WebError = FakeWebError
    module.SearXNGSearch = MagicMock(name="SearXNGSearch")
    module.WebClient = MagicMock(name="WebClient")
    return module


def result(title: str, url: str) -> MagicMock:
    item = MagicMock()
    item.to_dict.return_value = {"title": title, "url": url, "snippet": ""}
    return item


class SearxngUrlTests(unittest.TestCase):
    def test_config_wins_over_environment(self) -> None:
        with patch.dict(os.environ, {SEARXNG_URL_ENV: "http://env:8888"}):
            self.assertEqual(searxng_url({"searxng_url": "http://cfg:8888"}), "http://cfg:8888")
            self.assertEqual(searxng_url({}), "http://env:8888")

    def test_unset_means_web_is_disabled(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(searxng_url({}))
            self.assertIsNone(searxng_url({"searxng_url": "  "}))


class WebExecutorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.web = fake_web_module()
        self.fallback = MagicMock(return_value={"action": "SEND_MAIL"})
        with patch.dict("sys.modules", {"ethosoftlib.web": self.web}):
            self.executor = WebExecutor("http://localhost:8888", self.fallback)
        self.client = self.web.WebClient.return_value

    def test_uses_only_the_configured_searxng_instance(self) -> None:
        self.web.SearXNGSearch.assert_called_once_with("http://localhost:8888")
        self.web.WebClient.assert_called_once_with(provider=self.web.SearXNGSearch.return_value)

    def test_search_returns_results_and_caps_their_number(self) -> None:
        self.client.search.return_value = [result("Ankara", "https://a.example")]
        observation = self.executor("WEB_SEARCH", {"query": "ankara", "max_results": 50})
        self.client.search.assert_called_once_with("ankara", max_results=MAX_SEARCH_RESULTS)
        self.assertEqual(observation["status"], "success")
        self.assertEqual(observation["data"], {
            "query": "ankara",
            "results": [{"title": "Ankara", "url": "https://a.example", "snippet": ""}],
        })

    def test_open_limits_page_text(self) -> None:
        self.client.open_url.return_value.to_dict.return_value = {"url": "https://a.example"}
        observation = self.executor("WEB_OPEN", {"url": "https://a.example"})
        self.client.open_url.assert_called_once_with("https://a.example", max_chars=MAX_PAGE_CHARS)
        self.assertEqual(observation["data"], {"url": "https://a.example"})

    def test_network_failure_becomes_an_error_observation(self) -> None:
        self.client.search.side_effect = FakeWebError("HTTP 403 for http://localhost:8888/search")
        observation = self.executor("WEB_SEARCH", {"query": "ankara"})
        self.assertEqual(observation["status"], "error")
        self.assertIn("HTTP 403", observation["error"])

    def test_other_actions_go_to_the_configured_executor(self) -> None:
        self.assertEqual(self.executor("SEND_MAIL", {"to": "a@b.c"}), {"action": "SEND_MAIL"})
        self.fallback.assert_called_once_with("SEND_MAIL", {"to": "a@b.c"})


class BuildRuntimeWebTests(unittest.TestCase):
    def _runtime_info(self, pipeline_overrides: dict) -> tuple[dict, dict]:
        config = json.loads(json.dumps(cli.DEFAULT_CONFIG))
        config["pipeline"].update(pipeline_overrides)
        with patch.dict(os.environ, {}, clear=True), \
                patch.dict("sys.modules", {"ethosoftlib.web": fake_web_module()}):
            pipeline, _ = cli.build_runtime(config)
        return pipeline.tool_registry, pipeline.yargi._runtime_info

    def test_searxng_url_makes_web_tools_built_in(self) -> None:
        registry, info = self._runtime_info({"searxng_url": "http://localhost:8888"})
        for action in ("WEB_SEARCH", "WEB_OPEN"):
            self.assertEqual(registry[action]["_builtin_executor"], "web")
            self.assertNotIn("_executor_required", registry[action])
            self.assertTrue(info["tools"][action]["built_in"])
            self.assertFalse(info["tools"][action]["external_executor_needed"])
        # Downloading writes to disk and stays behind an explicit executor.
        self.assertFalse(info["tools"]["DOWNLOAD"]["built_in"])

    def test_without_searxng_web_tools_stay_unconfigured(self) -> None:
        registry, info = self._runtime_info({})
        self.assertNotIn("_builtin_executor", registry["WEB_SEARCH"])
        self.assertFalse(info["tools"]["WEB_SEARCH"]["built_in"])


if __name__ == "__main__":
    unittest.main()
