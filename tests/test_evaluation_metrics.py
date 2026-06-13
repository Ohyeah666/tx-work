import numpy as np

from evaluation_metrics import BacktrackingMetricTracker, BACKTRACK_METRIC_NAMES


def test_backtracking_metric_tracker_records_all_metrics():
    tracker = BacktrackingMetricTracker(
        memory_threshold=0.5,
        future_window=2,
        gain_threshold=1.0,
        oscillation_windows=(4, 6),
    )
    tracker.reset(start_index=0)
    tracker.record_step(next_index=1, distance=1.0, next_memory=0.1, new_area_gain=10.0)
    tracker.record_step(next_index=0, distance=1.0, next_memory=0.9, new_area_gain=0.0)
    tracker.record_step(next_index=2, distance=2.0, next_memory=0.9, new_area_gain=5.0)

    metrics = tracker.compute(total_travel_distance=4.0)

    assert set(BACKTRACK_METRIC_NAMES).issubset(metrics.keys())
    np.testing.assert_allclose(metrics['immediate_reverse_rate'], 1 / 3)
    np.testing.assert_allclose(metrics['revisited_node_ratio'], 1 / 3)
    np.testing.assert_allclose(metrics['revisited_distance_ratio'], 1 / 4)
    np.testing.assert_allclose(metrics['local_oscillation_count_k4'], 1)
    np.testing.assert_allclose(metrics['local_oscillation_count_k6'], 1)
    np.testing.assert_allclose(metrics['nonproductive_backtrack_distance_ratio'], 0.0)
    np.testing.assert_allclose(metrics['necessary_backtrack_ratio'], 1.0)
    np.testing.assert_allclose(metrics['backtrack_productivity_score'], 10 / 3)
    np.testing.assert_allclose(metrics['unique_exploration_efficiency'], 15 / 4)
