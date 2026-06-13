import numpy as np

from parameter import (
    BACKTRACK_FUTURE_WINDOW,
    BACKTRACK_GAIN_THRESHOLD,
    BACKTRACK_MEMORY_THRESHOLD,
    LOCAL_OSCILLATION_WINDOWS,
)


BACKTRACK_METRIC_NAMES = [
    'immediate_reverse_rate',
    'revisited_node_ratio',
    'revisited_distance_ratio',
    'local_oscillation_count_k4',
    'local_oscillation_count_k6',
    'nonproductive_backtrack_distance_ratio',
    'necessary_backtrack_ratio',
    'backtrack_productivity_score',
    'unique_exploration_efficiency',
]


class BacktrackingMetricTracker:
    def __init__(
        self,
        memory_threshold=BACKTRACK_MEMORY_THRESHOLD,
        future_window=BACKTRACK_FUTURE_WINDOW,
        gain_threshold=BACKTRACK_GAIN_THRESHOLD,
        oscillation_windows=LOCAL_OSCILLATION_WINDOWS,
    ):
        self.memory_threshold = memory_threshold
        self.future_window = future_window
        self.gain_threshold = gain_threshold
        self.oscillation_windows = tuple(oscillation_windows)
        self.node_history = []
        self.records = []

    def reset(self, start_index):
        self.node_history = [int(start_index)]
        self.records = []

    def record_step(self, next_index, distance, next_memory, new_area_gain):
        if not self.node_history:
            self.reset(next_index)

        next_index = int(next_index)
        previous_index = self.node_history[-2] if len(self.node_history) >= 2 else None
        visited_before = next_index in self.node_history
        immediate_reverse = previous_index is not None and next_index == previous_index
        local_oscillation = {
            window: next_index in self.node_history[-window:]
            for window in self.oscillation_windows
        }

        self.records.append({
            'next_index': next_index,
            'distance': float(distance),
            'next_memory': float(next_memory),
            'new_area_gain': float(max(new_area_gain, 0)),
            'visited_before': bool(visited_before),
            'immediate_reverse': bool(immediate_reverse),
            'local_oscillation': local_oscillation,
        })
        self.node_history.append(next_index)

    def compute(self, total_travel_distance=None):
        total_steps = len(self.records)
        if total_steps == 0:
            return {name: 0.0 for name in BACKTRACK_METRIC_NAMES}

        distances = np.array([record['distance'] for record in self.records], dtype=float)
        new_area_gains = np.array([record['new_area_gain'] for record in self.records], dtype=float)
        total_distance = float(np.sum(distances) if total_travel_distance is None else total_travel_distance)
        total_distance = max(total_distance, 1e-6)

        immediate_reverse = np.array([record['immediate_reverse'] for record in self.records], dtype=bool)
        revisited = np.array([record['visited_before'] for record in self.records], dtype=bool)
        backtrack = np.array(
            [record['next_memory'] > self.memory_threshold for record in self.records],
            dtype=bool,
        )
        future_gains = self._future_gains(new_area_gains)
        productive_backtrack = backtrack & (future_gains >= self.gain_threshold)
        nonproductive_backtrack = backtrack & (future_gains < self.gain_threshold)
        backtrack_distance = float(np.sum(distances[backtrack]))

        metrics = {
            'immediate_reverse_rate': float(np.mean(immediate_reverse)),
            'revisited_node_ratio': float(np.mean(revisited)),
            'revisited_distance_ratio': float(np.sum(distances[revisited]) / total_distance),
            'local_oscillation_count_k4': float(self._local_oscillation_count(4)),
            'local_oscillation_count_k6': float(self._local_oscillation_count(6)),
            'nonproductive_backtrack_distance_ratio': float(
                np.sum(distances[nonproductive_backtrack]) / total_distance
            ),
            'necessary_backtrack_ratio': float(
                np.sum(distances[productive_backtrack]) / max(backtrack_distance, 1e-6)
            ),
            'backtrack_productivity_score': float(
                np.sum(future_gains[backtrack]) / max(backtrack_distance, 1e-6)
            ),
            'unique_exploration_efficiency': float(np.sum(new_area_gains) / total_distance),
        }
        return metrics

    def _future_gains(self, new_area_gains):
        future_gains = np.zeros_like(new_area_gains, dtype=float)
        for index in range(new_area_gains.shape[0]):
            end = min(new_area_gains.shape[0], index + self.future_window)
            future_gains[index] = np.sum(new_area_gains[index:end])
        return future_gains

    def _local_oscillation_count(self, window):
        if window not in self.oscillation_windows:
            return 0

        return sum(
            record['local_oscillation'].get(window, False)
            for record in self.records
        )
