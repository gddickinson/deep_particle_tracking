"""MetricsCalculator regressions."""
import numpy as np

from prediction.metrics import MetricsCalculator as M


def test_detection_matching_maximises_true_positives():
    # Unthresholded min-cost matching paired (11, 4) with a far prediction: 2 TP instead of 3.
    gt = np.array([[6.0, 11.0], [11.0, 4.0], [7.0, 4.0]])
    pred = np.array([[7.0, 4.0], [5.0, 11.0], [3.0, 7.0]])
    m = M.calculate_detection_metrics(pred, gt, distance_threshold=5.0)
    assert (m['tp'], m['fp'], m['fn']) == (3, 0, 0)


def test_detection_simple_counts():
    gt = np.array([[0.0, 0.0], [10.0, 10.0]])
    pred = np.array([[0.5, 0.0], [30.0, 30.0]])
    m = M.calculate_detection_metrics(pred, gt, distance_threshold=2.0)
    assert (m['tp'], m['fp'], m['fn']) == (1, 1, 1)


def test_id_switch_counted_across_a_gap():
    # GT 0 visible in frames 0, 1, 3 (blinks off in 2) while GT 1 stays visible; the prediction for
    # GT 0 changes identity after the gap. CLEAR MOT compares with the last correspondence.
    nan = np.nan
    far = [[20, 20], [20, 21], [20, 22], [20, 23]]
    gt = [np.array([[5, 5], [5, 6], [nan, nan], [5, 7]], float), np.array(far, float)]
    pred = [np.array([[5, 5], [5, 6], [nan, nan], [nan, nan]], float),
            np.array([[nan, nan], [nan, nan], [nan, nan], [5, 7]], float), np.array(far, float)]
    m = M.calculate_tracking_metrics(pred, gt, distance_threshold=2.0)
    assert m['id_switches'] == 1
    assert m['tp'] == 7 and m['fp'] == 0 and m['fn'] == 0


def test_no_switch_for_consistent_identity():
    gt = [np.array([[5, 5], [5, 6], [5, 7]], float)]
    pred = [np.array([[5, 5], [5, 6], [5, 7]], float)]
    m = M.calculate_tracking_metrics(pred, gt, distance_threshold=2.0)
    assert m['id_switches'] == 0 and m['mota'] == 1.0
