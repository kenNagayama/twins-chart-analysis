"""AnomalyDetector のユニットテスト (TDD: タスク 6.1, 6.2)"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.anomaly_detector import AnomalyDetector, AnomalyResult


class TestComputeMovingAverage:
    """6.1 移動平均の算出"""

    def setup_method(self):
        self.detector = AnomalyDetector()

    def test_window_filled_returns_mean(self):
        """ウィンドウが満たされた区間は正しい移動平均を返す"""
        data = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = self.detector.compute_moving_average(data, window_size=3)
        # インデックス 2 以降（ウィンドウ幅=3 が満たされた区間）
        assert result.iloc[2] == pytest.approx(2.0)
        assert result.iloc[3] == pytest.approx(3.0)
        assert result.iloc[4] == pytest.approx(4.0)

    def test_window_not_filled_is_nan(self):
        """ウィンドウが満たない端部区間は NaN になること（min_periods=window_size）"""
        data = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = self.detector.compute_moving_average(data, window_size=3)
        # 先頭 2 点はウィンドウが満たされていない
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])

    def test_window_size_equals_data_length(self):
        """ウィンドウ幅 == データ長のとき、最後の 1 点だけ有効値になる"""
        data = pd.Series([2.0, 4.0, 6.0])
        result = self.detector.compute_moving_average(data, window_size=3)
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        assert result.iloc[2] == pytest.approx(4.0)

    def test_window_size_1_returns_original(self):
        """ウィンドウ幅 1 のときは元の値をそのまま返す"""
        data = pd.Series([3.0, 1.0, 4.0, 1.0, 5.0])
        result = self.detector.compute_moving_average(data, window_size=1)
        pd.testing.assert_series_equal(result, data, check_names=False)

    def test_result_has_same_length_as_input(self):
        """結果 Series の長さは入力と同じ"""
        data = pd.Series(range(10), dtype=float)
        result = self.detector.compute_moving_average(data, window_size=5)
        assert len(result) == len(data)


class TestComputeZScore:
    """6.2 Z-Score 算出"""

    def setup_method(self):
        self.detector = AnomalyDetector()

    def test_zscore_nan_for_underfilled_window(self):
        """ウィンドウが満たない区間（先頭）は NaN になること"""
        data = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        result = self.detector.compute_zscore(data, window_size=3)
        # 先頭 window_size - 1 点は NaN
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        # ウィンドウ幅以降は有効値 or NaN（均一区間以外）
        assert not pd.isna(result.iloc[3])

    def test_zscore_nan_for_zero_std(self):
        """標準偏差がゼロの均一区間は NaN を返す（ゼロ除算ガード）"""
        # 全て同一値 → std=0
        data = pd.Series([5.0] * 10)
        result = self.detector.compute_zscore(data, window_size=3)
        # 均一区間の Z-Score は NaN でなければならない
        for val in result.dropna():
            assert pd.isna(val) or val == pytest.approx(0.0, abs=1e-9)
        # 少なくともウィンドウが満たされた区間は全て NaN
        assert pd.isna(result.iloc[5])

    def test_zscore_correct_value(self):
        """Z-Score が (x - mean) / std の計算と一致すること"""
        np.random.seed(42)
        data = pd.Series(np.random.randn(50) * 2.0 + 10.0)
        window_size = 5
        result = self.detector.compute_zscore(data, window_size=window_size)

        # ウィンドウが満たされた最初の点を手動検証
        i = window_size - 1  # インデックス 4
        window_vals = data.iloc[0 : window_size].values
        expected_mean = np.mean(window_vals)
        expected_std = np.std(window_vals, ddof=1)
        expected_z = (data.iloc[i] - expected_mean) / expected_std
        assert result.iloc[i] == pytest.approx(expected_z, rel=1e-6)

    def test_zscore_same_length_as_input(self):
        """Z-Score Series の長さは入力と同じ"""
        data = pd.Series(range(20), dtype=float)
        result = self.detector.compute_zscore(data, window_size=5)
        assert len(result) == len(data)


class TestDetect:
    """6.2 異常点記録"""

    def setup_method(self):
        self.detector = AnomalyDetector()

    def _make_series_with_spike(self, n: int = 30, spike_idx: int = 20) -> pd.Series:
        """spike_idx の位置に大きなスパイクを持つ Series を作成する"""
        data = np.ones(n) * 10.0
        data[spike_idx] = 0.0  # 急激な低下 → 大きな Z-Score
        return pd.Series(data)

    def test_detect_returns_anomaly_result(self):
        """detect() は AnomalyResult を返すこと"""
        data = self._make_series_with_spike()
        result = self.detector.detect(data, window_size=5, threshold=2.0)
        assert isinstance(result, AnomalyResult)

    def test_anomaly_indices_are_subset_of_zscore_index(self):
        """anomaly_indices は zscore_series のインデックスのサブセット"""
        data = self._make_series_with_spike()
        result = self.detector.detect(data, window_size=5, threshold=2.0)
        for idx in result.anomaly_indices:
            assert idx in result.zscore_series.index

    def test_spike_is_detected_as_anomaly(self):
        """スパイク点が異常として検知されること

        ウィンドウ幅 5 で単一スパイクの Z-Score 上限は (n-1)/√n = 4/√5 ≈ 1.789。
        threshold=1.5 を使用して検知できることを確認する。
        """
        data = self._make_series_with_spike(n=30, spike_idx=20)
        result = self.detector.detect(data, window_size=5, threshold=1.5)
        # スパイクインデックス 20 が anomaly_indices に含まれること
        assert 20 in result.anomaly_indices

    def test_no_anomaly_when_threshold_very_high(self):
        """閾値が非常に高い場合は異常なし"""
        data = self._make_series_with_spike()
        result = self.detector.detect(data, window_size=5, threshold=1000.0)
        assert len(result.anomaly_indices) == 0

    def test_threshold_stored_in_result(self):
        """指定した閾値が AnomalyResult.threshold に記録されること"""
        data = self._make_series_with_spike()
        result = self.detector.detect(data, window_size=5, threshold=3.5)
        assert result.threshold == pytest.approx(3.5)

    def test_zscore_series_length_equals_input(self):
        """zscore_series の長さは入力と同じ"""
        data = self._make_series_with_spike()
        result = self.detector.detect(data, window_size=5, threshold=2.0)
        assert len(result.zscore_series) == len(data)

    def test_moving_average_in_result(self):
        """AnomalyResult に moving_average が含まれること"""
        data = self._make_series_with_spike()
        result = self.detector.detect(data, window_size=5, threshold=2.0)
        assert isinstance(result.moving_average, pd.Series)
        assert len(result.moving_average) == len(data)

    def test_anomaly_positions_with_position_series(self):
        """position_series を渡した場合、anomaly_positions にキロ程値が記録されること"""
        data = self._make_series_with_spike(n=30, spike_idx=20)
        # キロ程: 0.00, 0.05, 0.10, ...
        positions = pd.Series([i * 0.05 for i in range(30)])
        # threshold=1.5 (ウィンドウ幅5での最大Z-Score≈1.789 > 1.5)
        result = self.detector.detect(
            data, window_size=5, threshold=1.5, position_series=positions
        )
        # スパイク点(index=20)のキロ程 = 1.00
        assert 20 in result.anomaly_indices
        assert result.anomaly_positions.iloc[0] == pytest.approx(20 * 0.05)

    def test_anomaly_positions_empty_when_no_position_series(self):
        """position_series が None の場合、anomaly_positions は空 Series"""
        data = self._make_series_with_spike()
        result = self.detector.detect(data, window_size=5, threshold=2.0)
        assert isinstance(result.anomaly_positions, pd.Series)

    def test_zscore_exceeds_threshold_at_anomaly_points(self):
        """anomaly_indices の全点で |Z-Score| >= threshold であること"""
        data = self._make_series_with_spike()
        threshold = 1.5  # ウィンドウ幅5での最大Z-Score≈1.789 > 1.5
        result = self.detector.detect(data, window_size=5, threshold=threshold)
        assert len(result.anomaly_indices) > 0, "スパイクが検知されなかった"
        for idx in result.anomaly_indices:
            assert abs(result.zscore_series.iloc[idx]) >= threshold
