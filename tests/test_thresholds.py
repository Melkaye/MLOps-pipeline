from train import check_thresholds


def test_thresholds_pass_when_all_metrics_meet_minimums():
    metrics = {"accuracy": 0.85, "f1_score": 0.80, "auc_roc": 0.90}
    assert check_thresholds(metrics, {"accuracy": 0.70, "auc_roc": 0.80}) == []


def test_thresholds_fail_when_a_metric_is_too_low():
    failures = check_thresholds({"accuracy": 0.60}, {"accuracy": 0.70})
    assert len(failures) == 1
    assert "accuracy" in failures[0]


def test_thresholds_fail_when_metric_is_missing():
    failures = check_thresholds({"accuracy": 0.90}, {"auc_roc": 0.80})
    assert len(failures) == 1
    assert "not logged" in failures[0]
