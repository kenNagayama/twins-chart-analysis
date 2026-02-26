"""ProcessingLayer: 移動中央値・SGフィルタ"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from scipy.signal import savgol_filter


@dataclass(frozen=True)
class FilterConfig:
    """フィルタ設定値オブジェクト。

    Attributes:
        median_enabled: 移動中央値フィルタの ON/OFF フラグ
        median_window: 移動中央値フィルタのウィンドウ幅
        savgol_enabled: Savitzky-Golay フィルタの ON/OFF フラグ
        savgol_window: SG フィルタのウィンドウ幅（必ず奇数・ParameterValidator 補正済み）
        savgol_polyorder: SG フィルタの多項式次数（デフォルト 2）
    """

    median_enabled: bool = True
    median_window: int = 5
    savgol_enabled: bool = True
    savgol_window: int = 11   # 必ず奇数（ParameterValidator 補正済み）
    savgol_polyorder: int = 2


@dataclass(frozen=True)
class FilterResult:
    """フィルタ処理結果値オブジェクト。

    Attributes:
        original: フィルタ適用前の元データ Series
        filtered: フィルタ適用後のデータ Series
        config: 使用した FilterConfig
    """

    original: pd.Series
    filtered: pd.Series
    config: FilterConfig


class NoiseFilter:
    """移動中央値フィルタおよび Savitzky-Golay フィルタを ON/OFF 制御付きで適用するクラス。

    フィルタ適用順序: 移動中央値 → Savitzky-Golay
    両フィルタが OFF の場合は入力 Series をそのまま返す。
    状態を持たず、純粋関数的に動作する。
    """

    def apply(self, series: pd.Series, config: FilterConfig) -> FilterResult:
        """FilterConfig に従いフィルタを適用する。

        適用順序:
            1. median_enabled=True の場合: 移動中央値フィルタを適用
            2. savgol_enabled=True の場合: Savitzky-Golay フィルタを適用
        両方 OFF の場合は original == filtered となる。

        Args:
            series: フィルタ対象の摩耗値 Series
            config: フィルタ設定（ON/OFF フラグ・パラメータ）

        Returns:
            FilterResult（元データと処理後データを保持）
        """
        current = series

        # ステップ1: 移動中央値フィルタ（有効な場合）
        if config.median_enabled:
            current = self.apply_rolling_median(current, window=config.median_window)

        # ステップ2: Savitzky-Golay フィルタ（有効な場合）
        if config.savgol_enabled:
            current = self.apply_savgol(
                current,
                window=config.savgol_window,
                polyorder=config.savgol_polyorder,
            )

        return FilterResult(
            original=series,
            filtered=current,
            config=config,
        )

    def apply_rolling_median(
        self, series: pd.Series, window: int
    ) -> pd.Series:
        """移動中央値フィルタを適用する。

        pandas の rolling API を使い、指定ウィンドウ幅の移動中央値を計算する。
        min_periods=1, center=True を設定することで、データ端部でも NaN が発生せず
        連続した波形を返す。

        Args:
            series: フィルタ対象の Series
            window: 移動中央値のウィンドウ幅

        Returns:
            移動中央値フィルタ適用後の Series（インデックスは入力を保持）
        """
        result = series.rolling(window=window, min_periods=1, center=True).median()
        return result

    def apply_savgol(
        self,
        series: pd.Series,
        window: int,
        polyorder: int,
    ) -> pd.Series:
        """Savitzky-Golay フィルタを適用する。

        scipy.signal.savgol_filter を使って指定ウィンドウ幅・多項式次数でフィルタリングする。
        ウィンドウ幅は ParameterValidator で奇数補正済みの値を受け取ることを前提とする。
        呼び出し前に window > polyorder であることを確認し、条件違反時は ValueError を送出する。

        Args:
            series: フィルタ対象の Series
            window: SG フィルタのウィンドウ幅（奇数・補正済み前提）
            polyorder: SG フィルタの多項式次数

        Returns:
            Savitzky-Golay フィルタ適用後の Series（インデックスは入力を保持）

        Raises:
            ValueError: window <= polyorder の場合
        """
        if window <= polyorder:
            raise ValueError(
                f"window ({window}) は polyorder ({polyorder}) より大きくなければなりません。"
                f"window > polyorder の条件を満たすように設定してください。"
            )

        filtered_values = savgol_filter(series.to_numpy(), window_length=window, polyorder=polyorder)
        return pd.Series(filtered_values, index=series.index, name=series.name)
