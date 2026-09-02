from securescan_ml.labels import (
    ID_TO_LABEL,
    LABEL_ORDER,
    LABEL_TO_ID,
    NUM_LABELS,
    REMEDIATIONS,
    SEVERITY,
    Label,
    Severity,
    label_from_cwe,
    severity_for,
)


def test_safe_is_class_zero():
    """metrics.py assumes SAFE is index 0 for FPR and detection rate."""
    assert LABEL_TO_ID[Label.SAFE] == 0


def test_id_mapping_round_trips():
    for label in LABEL_ORDER:
        assert ID_TO_LABEL[LABEL_TO_ID[label]] is label
    assert len(Label) == NUM_LABELS


def test_every_label_has_a_severity():
    for label in Label:
        assert label in SEVERITY


def test_only_safe_has_no_severity():
    assert severity_for(Label.SAFE) is Severity.NONE
    for label in Label:
        if label is not Label.SAFE:
            assert severity_for(label) is not Severity.NONE


def test_every_vulnerable_label_has_remediation():
    """A finding with no suggested fix is a dead end for the developer."""
    for label in Label:
        if label is not Label.SAFE:
            assert label in REMEDIATIONS, f"{label.value} has no remediation template"


def test_cwe_mapping():
    assert label_from_cwe("CWE-89") is Label.SQL_INJECTION
    assert label_from_cwe("cwe-79") is Label.XSS
    assert label_from_cwe("CWE-99999") is None
