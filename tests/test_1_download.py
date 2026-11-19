#!/usr/bin/env python3

"""Download unit tests."""

from pathlib import Path

from implicit_bathy.download.automatic_download_dsm import download_dsm
from implicit_bathy.download.automatic_download_s2 import download_s2_tif
from implicit_bathy.download.automatic_download_swot import download_swot


def test_download_dsm(path_tests, tmp_path):

    path_dsm = Path(tmp_path) / "output" / "dsm" / "COP30.tif"
    if path_dsm.exists():
        path_dsm.unlink()
    download_dsm(path_tests)

    assert path_dsm.exists()


def test_download_s2(path_tests, tmp_path):

    path_s2 = Path(tmp_path) / "output" / "data" / "S2.tif"
    if path_s2.exists():
        path_s2.unlink()
    download_s2_tif(path_tests)

    assert path_s2.exists()


def test_download_swot(path_tests, tmp_path):

    for f in (Path(tmp_path) / "output" / "inputs_nc").glob("*.nc"):
        f.unlink()

    download_swot(path_tests)
    assert len(list((Path(tmp_path) / "output" / "inputs_nc").glob("*.nc"))) > 0
