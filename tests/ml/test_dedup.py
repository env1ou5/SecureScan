from securescan_ml.data.dedup import deduplicate, jaccard, minhash
from securescan_ml.data.schema import VulnRecord
from securescan_ml.labels import Label


def record(code: str, source: str = "mined") -> VulnRecord:
    return VulnRecord("repo", "a.py", "python", code, Label.SAFE, 1, 2, source, source == "juliet")


def test_identical_code_has_identical_signature():
    code = "def f(x):\n    return x + 1\n"
    assert minhash(code) == minhash(code)


def test_jaccard_reflects_similarity():
    a = minhash("def f(x):\n    return x + 1\n")
    b = minhash("def f(x):\n    return x + 1  # comment\n")
    c = minhash("class Wholly:\n    different = True\n")
    assert jaccard(a, b) > jaccard(a, c)


def test_duplicates_are_removed():
    code = "def handler(request):\n    return process(request.data)\n"
    records = [record(code) for _ in range(5)] + [record("def other():\n    pass\n")]
    kept, dropped = deduplicate(records)
    assert len(kept) == 2
    assert len(dropped) == 4


def test_real_code_survives_over_synthetic():
    code = "def q(i):\n    cursor.execute('SELECT * FROM t WHERE id=' + i)\n"
    records = [record(code, "juliet"), record(code, "mined")]
    kept, _ = deduplicate(records)
    assert len(kept) == 1
    assert kept[0].source == "mined"
