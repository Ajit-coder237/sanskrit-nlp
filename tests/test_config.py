"""Tests for the config loader."""

import textwrap

from sanskrit_nlp.utils.config import Config, load_config


def test_load_config(tmp_path):
    path = tmp_path / "base.yaml"
    path.write_text(
        textwrap.dedent(
            """
            project: sanskrit-nlp
            ocr:
              metrics: [cer, wer]
            """
        ),
        encoding="utf-8",
    )
    cfg = load_config(path)
    assert cfg["project"] == "sanskrit-nlp"
    assert cfg.get("ocr.metrics") == ["cer", "wer"]
    assert cfg.get("missing.key", "default") == "default"


def test_config_from_dict():
    cfg = Config.from_dict({"a": {"b": 1}})
    assert cfg.get("a.b") == 1
    cfg.set("a.c", 2)
    assert cfg.get("a.c") == 2
