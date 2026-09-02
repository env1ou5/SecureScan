import pytest
from securescan_ml.data.schema import VulnRecord
from securescan_ml.data.splits import class_weights, make_splits
from securescan_ml.labels import LABEL_ORDER, Label


def record(repo: str, source: str = "mined", label: Label = Label.SAFE) -> VulnRecord:
    return VulnRecord(
        repository_id=repo,
        file_path="a.py",
        language="python",
        code=f"def f_{repo}():\n    pass\n",
        vulnerability_type=label,
        start_line=1,
        end_line=2,
        source=source,
        is_synthetic=(source == "juliet"),
    )


def test_repository_never_straddles_splits():
    records = [record(f"repo{i}") for i in range(200) for _ in range(3)]
    splits = make_splits(records)
    for name_a, name_b in (
        ("train", "validation"),
        ("train", "test_full"),
        ("validation", "test_full"),
    ):
        repos_a = {r.repository_id for r in getattr(splits, name_a)}
        repos_b = {r.repository_id for r in getattr(splits, name_b)}
        assert not (repos_a & repos_b), f"{name_a} and {name_b} share repositories"


def test_real_test_set_excludes_synthetic():
    records = [record(f"r{i}", "juliet", Label.XSS) for i in range(100)]
    records += [record(f"m{i}", "mined") for i in range(100)]
    splits = make_splits(records)
    assert all(not r.is_synthetic for r in splits.test_real)
    assert all(r.source == "mined" for r in splits.test_real)


def test_splits_are_deterministic():
    records = [record(f"repo{i}") for i in range(100)]
    a = make_splits(records)
    b = make_splits(list(reversed(records)))
    assert {r.repository_id for r in a.train} == {r.repository_id for r in b.train}


def test_is_synthetic_must_match_source():
    with pytest.raises(ValueError, match="contradicts source"):
        VulnRecord("r", "a.py", "python", "x=1", Label.SAFE, 1, 1, "mined", True)


def test_class_weights_favor_rare_classes():
    records = [record(f"r{i}") for i in range(90)]
    records += [record(f"s{i}", label=Label.SQL_INJECTION) for i in range(10)]
    weights = class_weights(records, LABEL_ORDER)
    safe_w = weights[LABEL_ORDER.index(Label.SAFE)]
    sqli_w = weights[LABEL_ORDER.index(Label.SQL_INJECTION)]
    assert sqli_w > safe_w
