import tempfile
from pathlib import Path

import pytest
import yaml

from config import load_config, load_named, require


def test_load_config_defaults_to_empty_dict_when_file_missing():
    assert load_config(Path("does_not_exist.yaml")) == {}


def test_load_config_reads_yaml():
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump({"risk": {"max_loss_pct": 0.01}}, f)
        name = f.name
    assert load_config(Path(name))["risk"]["max_loss_pct"] == 0.01


def test_project_configs_all_load():
    for name in ("data", "strategy", "model", "risk"):
        assert load_named(name), f"configs/{name}.yaml is missing or empty"


def test_require_reads_nested_keys():
    assert require({"mlp": {"patience": 10}}, "mlp.patience") == 10


def test_require_raises_rather_than_defaulting_silently():
    with pytest.raises(KeyError):
        require({"a": 1}, "b")
    with pytest.raises(KeyError):
        require({"a": {"b": 1}}, "a.c")
