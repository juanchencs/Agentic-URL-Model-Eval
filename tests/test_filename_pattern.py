from pathlib import Path
import re


PATTERN = re.compile(r"^(?P<data_source>.+)_(?P<scan_type>url|domain)_(?P<flag>clean|mal|unknown)\.csv$")


def test_dataset_filename_pattern():
    assert PATTERN.match("vt_url_clean.csv")
    assert PATTERN.match("vt_domain_mal.csv")
    assert not PATTERN.match("vt_url_clean.txt")
    assert not PATTERN.match("badname.csv")


def test_test_files_exist():
    root = Path(__file__).resolve().parent
    for name in [
        "vt_domain_clean.txt",
        "vt_url_clean.txt",
        "vt_domain_mal.txt",
        "vt_url_mal.txt",
    ]:
        assert (root / name).exists(), f"Missing test file: {name}"
