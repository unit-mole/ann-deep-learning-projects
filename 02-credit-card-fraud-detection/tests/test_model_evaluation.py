import numpy as np

from src.model_evaluation import calculate_metrics, select_decision_threshold


def test_calculate_metrics_returns_expected_confusion_matrix():
    y_true = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.8, 0.7, 0.2])

    metrics = calculate_metrics(y_true, probabilities, threshold=0.5)

    assert metrics["confusion_matrix"] == [[1, 1], [1, 1]]
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1_score"] == 0.5


def test_max_f1_threshold_is_valid():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    probabilities = np.array([0.01, 0.10, 0.25, 0.55, 0.80, 0.95])

    result = select_decision_threshold(
        y_true,
        probabilities,
        strategy="max_f1",
    )

    assert 0 <= result["threshold"] <= 1
    assert result["validation_f1"] > 0.9


def test_recall_target_strategy_meets_target_when_possible():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    probabilities = np.array([0.05, 0.15, 0.40, 0.45, 0.70, 0.90])

    result = select_decision_threshold(
        y_true,
        probabilities,
        strategy="recall_target",
        recall_target=0.66,
    )

    assert result["validation_recall"] >= 0.66
