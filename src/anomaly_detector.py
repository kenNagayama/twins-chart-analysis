"""ProcessingLayer: Z-Score異常検知"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AnomalyResult:
    """Z-Score 異常検知の結果値オブジェクト"""

    zscore_series: pd.Series  # 全点の Z-Score（NaN 含む）
    moving_average: pd.Series  # 移動平均
    anomaly_indices: pd.Index  # 閾値超過点のインデックス
    anomaly_positions: pd.Series  # 閾値超過点のキロ程値
    threshold: float


class AnomalyDetector:
    """Z-Score に基づく局所的異常摩耗点の検出・記録

    Requirements: 5.1, 5.2, 5.4, 5.6
    """

    def compute_moving_average(
        self,
        series: pd.Series,
        window_size: int,
    ) -> pd.Series:
        """移動平均を算出する。

        min_periods=window_size とすることでウィンドウが満たない区間は NaN になる。

        Requirements: 5.1
        """
        return series.rolling(window=window_size, min_periods=window_size).mean()

    def compute_zscore(
        self,
        series: pd.Series,
        window_size: int,
    ) -> pd.Series:
        """移動平均からの Z-Score を算出する。

        - ウィンドウが満たない区間は NaN（min_periods=window_size）
        - 標準偏差がゼロの均一区間では NaN を返す（ゼロ除算ガード）

        Requirements: 5.2, 5.6
        """
        rolling = series.rolling(window=window_size, min_periods=window_size)
        mean = rolling.mean()
        std = rolling.std()

        # std == 0 の区間はゼロ除算を防ぐため NaN にマスクする
        std_safe = std.where(std != 0, other=np.nan)

        return (series - mean) / std_safe

    def detect(
        self,
        series: pd.Series,
        window_size: int,
        threshold: float,
        position_series: pd.Series | None = None,
    ) -> AnomalyResult:
        """Z-Score が閾値を超えた点を異常箇所として返す。

        Parameters
        ----------
        series : pd.Series
            フィルタ処理後の摩耗測定値 Series
        window_size : int
            ローリングウィンドウ幅（ParameterValidator 検証済み）
        threshold : float
            異常判定 Z-Score 閾値（正の実数）
        position_series : pd.Series | None
            キロ程値の Series。指定した場合 AnomalyResult.anomaly_positions に格納する。

        Requirements: 5.4
        """
        moving_average = self.compute_moving_average(series, window_size)
        zscore_series = self.compute_zscore(series, window_size)

        # |Z-Score| >= threshold の点を異常として記録（NaN は除外）
        anomaly_mask = zscore_series.abs() >= threshold
        anomaly_indices = series.index[anomaly_mask.fillna(False)]

        if position_series is not None:
            anomaly_positions = position_series.loc[anomaly_indices].reset_index(
                drop=True
            )
        else:
            anomaly_positions = pd.Series(dtype=float)

        return AnomalyResult(
            zscore_series=zscore_series,
            moving_average=moving_average,
            anomaly_indices=anomaly_indices,
            anomaly_positions=anomaly_positions,
            threshold=threshold,
        )
