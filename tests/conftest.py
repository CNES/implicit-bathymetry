#!/usr/bin/env python3

import pytest
from pathlib import Path
import yaml


@pytest.fixture
def path_yaml(request):
    """Return path integration test"""
    return Path(request.config.getoption("--path_yaml"))


@pytest.fixture
def tmp_path(request):
    """Return path integration test"""
    return Path(request.config.getoption("--tmp_path"))


def rewrite_paths(path_yaml, old_prefix, tmp_path):
    """
    Need to change output path
    """

    with open(str(path_yaml)) as f:
        data = yaml.safe_load(f)

    for key, value in data.items():
        if isinstance(value, dict):
            for k, v in value.items():
                if isinstance(v, str) and v.startswith(old_prefix):
                    value[k] = v.replace(old_prefix, str(tmp_path))

        elif isinstance(value, str):
            if value.startswith(old_prefix):
                data[key] = value.replace(old_prefix, str(tmp_path))

    path_config = Path(tmp_path) / "output" / "cfg" / "config_global.yaml"
    with open(path_config, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return path_config


@pytest.fixture
def path_tests(path_yaml, tmp_path):
    path_tests_rewrite = rewrite_paths(path_yaml, "/data", tmp_path)
    return path_tests_rewrite


def pytest_addoption(parser):
    parser.addoption(
        "--path_yaml",
        action="store",
        help="path_yaml",
    )
    parser.addoption(
        "--tmp_path",
        action="store",
        help="tmp_path",
    )
