"""NoiseFilter のユニットテスト (タスク 4.1 / 4.2 / 4.3)"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.noise_filter import FilterConfig, FilterResult, NoiseFilter


# ---------------------------------------------------------------------------
# フィクスチャ
# ---------------------------------------------------------------------------


def make_noisy_series(n: int = 50, seed: int = 42) -> pd.Series:
    """ランダムノイズを含む摩耗値 Series を生成するヘルパー。"""
    rng = np.random.default_rng(seed)
    base = np.linspace(40.0, 38.0, n)
    noise = rng.normal(0, 0.5, n)
    return pd.Series(base + noise, name="摩耗_測定値")


def make_simple_series() -> pd.Series:
    """テスト用の単純な Series（検証しやすい値）。"""
    return pd.Series([1.0, 2.0, 10.0, 2.0, 1.0, 3.0, 2.0, 1.0], name="テスト")


# ---------------------------------------------------------------------------
# 4.1: 移動中央値フィルタの実装
# ---------------------------------------------------------------------------


class TestApplyRollingMedian:
    """移動中央値フィルタ（apply_rolling_median）の動作検証。"""

    def test_returns_series(self) -> None:
        """戻り値が pd.Series であることを確認する。"""
        nf = NoiseFilter()
        s = make_noisy_series()
        result = nf.apply_rolling_median(s, window=5)

        assert isinstance(result, pd.Series)

    def test_same_length_as_input(self) -> None:
        """出力 Series の長さは入力と同じである。"""
        nf = NoiseFilter()
        s = make_noisy_series(n=50)
        result = nf.apply_rolling_median(s, window=5)

        assert len(result) == len(s)

    def test_no_nan_with_min_periods_1(self) -> None:
        """min_periods=1 により端部でも NaN が生じない。"""
        nf = NoiseFilter()
        s = make_noisy_series(n=10)
        result = nf.apply_rolling_median(s, window=7)

        assert result.isna().sum() == 0, "min_periods=1 のため NaN が発生してはならない"

    def test_spike_removal(self) -> None:
        """スパイクノイズが中央値フィルタで平滑化される。"""
        nf = NoiseFilter()
        s = make_simple_series()  # [1, 2, 10, 2, 1, 3, 2, 1]
        result = nf.apply_rolling_median(s, window=3)

        # インデックス2の値（スパイク 10.0）が周辺の中央値に近い値になる
        assert result.iloc[2] < 10.0, "スパイクが平滑化されるべき"

    def test_center_true_behavior(self) -> None:
        """center=True により中央配置のウィンドウが使われる。"""
        nf = NoiseFilter()
        # 均一な値（中央値は同じ値になる）
        s = pd.Series([5.0] * 10)
        result = nf.apply_rolling_median(s, window=3)

        assert (result == 5.0).all(), "均一データでは中央値フィルタ後も同じ値になるべき"

    def test_window_1_returns_same_values(self) -> None:
        """window=1 の場合は元の値がそのまま返る。"""
        nf = NoiseFilter()
        s = make_simple_series()
        result = nf.apply_rolling_median(s, window=1)

        pd.testing.assert_series_equal(result, s, check_names=False)

    def test_result_index_preserved(self) -> None:
        """入力 Series のインデックスが保持される。"""
        nf = NoiseFilter()
        s = pd.Series([1.0, 2.0, 3.0], index=[10, 20, 30])
        result = nf.apply_rolling_median(s, window=3)

        assert list(result.index) == [10, 20, 30]


# ---------------------------------------------------------------------------
# 4.2: Savitzky-Golay フィルタの実装
# ---------------------------------------------------------------------------


class TestApplySavgol:
    """Savitzky-Golay フィルタ（apply_savgol）の動作検証。"""

    def test_returns_series(self) -> None:
        """戻り値が pd.Series であることを確認する。"""
        nf = NoiseFilter()
        s = make_noisy_series(n=50)
        result = nf.apply_savgol(s, window=11, polyorder=2)

        assert isinstance(result, pd.Series)

    def test_same_length_as_input(self) -> None:
        """出力 Series の長さは入力と同じである。"""
        nf = NoiseFilter()
        s = make_noisy_series(n=50)
        result = nf.apply_savgol(s, window=11, polyorder=2)

        assert len(result) == len(s)

    def test_smoothing_effect(self) -> None:
        """SG フィルタが平滑化効果を持つことを確認する。"""
        nf = NoiseFilter()
        s = make_noisy_series(n=100)
        result = nf.apply_savgol(s, window=11, polyorder=2)

        # フィルタ後の分散がフィルタ前より小さい（ノイズ除去）
        assert result.std() < s.std(), "SG フィルタ後の分散が元データより小さいべき"

    def test_window_must_be_odd(self) -> None:
        """ウィンドウ幅が奇数の場合は正常に動作する。"""
        nf = NoiseFilter()
        s = make_noisy_series(n=50)
        # 奇数ウィンドウ
        result = nf.apply_savgol(s, window=11, polyorder=2)

        assert len(result) == len(s)

    def test_window_less_than_polyorder_raises(self) -> None:
        """window <= polyorder の場合は ValueError が送出される。"""
        nf = NoiseFilter()
        s = make_noisy_series(n=50)

        with pytest.raises(ValueError, match="window"):
            nf.apply_savgol(s, window=3, polyorder=5)

    def test_window_equal_to_polyorder_raises(self) -> None:
        """window == polyorder の場合も ValueError が送出される。"""
        nf = NoiseFilter()
        s = make_noisy_series(n=50)

        with pytest.raises(ValueError):
            nf.apply_savgol(s, window=3, polyorder=3)

    def test_result_index_preserved(self) -> None:
        """入力 Series のインデックスが保持される。"""
        nf = NoiseFilter()
        s = pd.Series(np.linspace(1.0, 10.0, 20), index=range(100, 120))
        result = nf.apply_savgol(s, window=5, polyorder=2)

        assert list(result.index) == list(s.index)

    def test_minimum_valid_window(self) -> None:
        """最小有効パラメータ (window=3, polyorder=2) で動作する。"""
        nf = NoiseFilter()
        s = make_noisy_series(n=20)
        result = nf.apply_savgol(s, window=3, polyorder=2)

        assert len(result) == len(s)


# ---------------------------------------------------------------------------
# 4.3: フィルタ適用順序制御と ON/OFF 切り替えの実装
# ---------------------------------------------------------------------------


class TestApplyBothOff:
    """両フィルタ OFF の場合は元データをそのまま返す。"""

    def test_both_off_returns_filter_result(self) -> None:
        """戻り値が FilterResult であることを確認する。"""
        nf = NoiseFilter()
        s = make_simple_series()
        config = FilterConfig(median_enabled=False, savgol_enabled=False)
        result = nf.apply(s, config)

        assert isinstance(result, FilterResult)

    def test_both_off_original_equals_filtered(self) -> None:
        """両フィルタ OFF のとき、original と filtered が同じ値を持つ。"""
        nf = NoiseFilter()
        s = make_simple_series()
        config = FilterConfig(median_enabled=False, savgol_enabled=False)
        result = nf.apply(s, config)

        pd.testing.assert_series_equal(
            result.original,
            result.filtered,
            check_names=False,
        )

    def test_both_off_preserves_original(self) -> None:
        """元データが result.original に保持される。"""
        nf = NoiseFilter()
        s = make_simple_series()
        config = FilterConfig(median_enabled=False, savgol_enabled=False)
        result = nf.apply(s, config)

        pd.testing.assert_series_equal(result.original, s, check_names=False)


class TestApplyMedianOnly:
    """移動中央値フィルタのみ ON の場合のテスト。"""

    def test_median_only_returns_filter_result(self) -> None:
        nf = NoiseFilter()
        s = make_noisy_series()
        config = FilterConfig(median_enabled=True, savgol_enabled=False, median_window=5)
        result = nf.apply(s, config)

        assert isinstance(result, FilterResult)

    def test_median_only_filtered_differs_from_original(self) -> None:
        """中央値フィルタ適用後のデータは元データと異なる（ノイズ含む場合）。"""
        nf = NoiseFilter()
        s = make_noisy_series(n=100)
        config = FilterConfig(median_enabled=True, savgol_enabled=False, median_window=7)
        result = nf.apply(s, config)

        # 完全に一致することはない（ノイズが平滑化される）
        assert not result.original.equals(result.filtered), "フィルタ後のデータは元データと異なるべき"

    def test_median_only_preserves_original(self) -> None:
        """result.original は元データを保持する。"""
        nf = NoiseFilter()
        s = make_noisy_series()
        config = FilterConfig(median_enabled=True, savgol_enabled=False)
        result = nf.apply(s, config)

        pd.testing.assert_series_equal(result.original, s, check_names=False)

    def test_median_only_no_nan_in_filtered(self) -> None:
        """フィルタ後データに NaN が含まれない。"""
        nf = NoiseFilter()
        s = make_noisy_series(n=30)
        config = FilterConfig(median_enabled=True, savgol_enabled=False, median_window=5)
        result = nf.apply(s, config)

        assert result.filtered.isna().sum() == 0


class TestApplySavgolOnly:
    """Savitzky-Golay フィルタのみ ON の場合のテスト。"""

    def test_savgol_only_returns_filter_result(self) -> None:
        nf = NoiseFilter()
        s = make_noisy_series(n=50)
        config = FilterConfig(median_enabled=False, savgol_enabled=True, savgol_window=11, savgol_polyorder=2)
        result = nf.apply(s, config)

        assert isinstance(result, FilterResult)

    def test_savgol_only_preserves_original(self) -> None:
        """result.original は元データを保持する。"""
        nf = NoiseFilter()
        s = make_noisy_series(n=50)
        config = FilterConfig(median_enabled=False, savgol_enabled=True, savgol_window=11)
        result = nf.apply(s, config)

        pd.testing.assert_series_equal(result.original, s, check_names=False)

    def test_savgol_only_filtered_length_unchanged(self) -> None:
        """SG フィルタ後のデータ長が変わらない。"""
        nf = NoiseFilter()
        s = make_noisy_series(n=50)
        config = FilterConfig(median_enabled=False, savgol_enabled=True, savgol_window=11)
        result = nf.apply(s, config)

        assert len(result.filtered) == len(s)


class TestApplyBothOn:
    """両フィルタ ON の場合のテスト（移動中央値 → SG の順）。"""

    def test_both_on_returns_filter_result(self) -> None:
        nf = NoiseFilter()
        s = make_noisy_series(n=100)
        config = FilterConfig(
            median_enabled=True,
            savgol_enabled=True,
            median_window=5,
            savgol_window=11,
            savgol_polyorder=2,
        )
        result = nf.apply(s, config)

        assert isinstance(result, FilterResult)

    def test_both_on_preserves_original(self) -> None:
        """元データは変更されず result.original に保持される。"""
        nf = NoiseFilter()
        s = make_noisy_series(n=100)
        config = FilterConfig(median_enabled=True, savgol_enabled=True)
        result = nf.apply(s, config)

        pd.testing.assert_series_equal(result.original, s, check_names=False)

    def test_both_on_applies_median_then_savgol(self) -> None:
        """両方 ON では移動中央値 → SG の順で適用される。

        検証方法: 両方 ON の結果が、中央値のみ適用したデータに SG を適用した結果と等しい。
        """
        nf = NoiseFilter()
        s = make_noisy_series(n=100)
        median_w = 5
        savgol_w = 11
        polyorder = 2

        # 両方 ON で apply を呼ぶ
        config_both = FilterConfig(
            median_enabled=True,
            savgol_enabled=True,
            median_window=median_w,
            savgol_window=savgol_w,
            savgol_polyorder=polyorder,
        )
        result_both = nf.apply(s, config_both)

        # 手動で順序通りに適用した場合
        after_median = nf.apply_rolling_median(s, window=median_w)
        expected = nf.apply_savgol(after_median, window=savgol_w, polyorder=polyorder)

        pd.testing.assert_series_equal(result_both.filtered, expected, check_names=False)

    def test_both_on_filtered_length_unchanged(self) -> None:
        """両フィルタ適用後もデータ長が変わらない。"""
        nf = NoiseFilter()
        s = make_noisy_series(n=100)
        config = FilterConfig(median_enabled=True, savgol_enabled=True)
        result = nf.apply(s, config)

        assert len(result.filtered) == len(s)

    def test_both_on_config_preserved(self) -> None:
        """使用した FilterConfig が result.config に保持される。"""
        nf = NoiseFilter()
        s = make_noisy_series(n=50)
        config = FilterConfig(
            median_enabled=True,
            savgol_enabled=True,
            median_window=5,
            savgol_window=11,
            savgol_polyorder=2,
        )
        result = nf.apply(s, config)

        assert result.config == config


# ---------------------------------------------------------------------------
# FilterConfig データクラスの構造確認
# ---------------------------------------------------------------------------


class TestFilterConfigDataclass:
    """FilterConfig データクラスの構造と不変条件を検証する。"""

    def test_is_frozen(self) -> None:
        config = FilterConfig()
        with pytest.raises(Exception):
            config.median_enabled = False  # type: ignore[misc]

    def test_default_values(self) -> None:
        """デフォルト値の確認。"""
        config = FilterConfig()
        assert config.median_enabled is True
        assert config.savgol_enabled is True
        assert config.median_window == 5
        assert config.savgol_window == 11
        assert config.savgol_polyorder == 2

    def test_custom_values(self) -> None:
        config = FilterConfig(
            median_enabled=False,
            savgol_enabled=True,
            median_window=7,
            savgol_window=13,
            savgol_polyorder=3,
        )
        assert config.median_enabled is False
        assert config.savgol_enabled is True
        assert config.median_window == 7
        assert config.savgol_window == 13
        assert config.savgol_polyorder == 3


# ---------------------------------------------------------------------------
# FilterResult データクラスの構造確認
# ---------------------------------------------------------------------------


class TestFilterResultDataclass:
    """FilterResult データクラスの構造を検証する。"""

    def test_is_frozen(self) -> None:
        s = pd.Series([1.0, 2.0, 3.0])
        config = FilterConfig()
        result = FilterResult(original=s, filtered=s, config=config)
        with pytest.raises(Exception):
            result.original = pd.Series([4.0])  # type: ignore[misc]

    def test_fields_accessible(self) -> None:
        """全フィールドにアクセスできる。"""
        s_orig = pd.Series([1.0, 2.0, 3.0])
        s_filt = pd.Series([1.1, 1.9, 3.1])
        config = FilterConfig(median_enabled=True, savgol_enabled=False)
        result = FilterResult(original=s_orig, filtered=s_filt, config=config)

        pd.testing.assert_series_equal(result.original, s_orig)
        pd.testing.assert_series_equal(result.filtered, s_filt)
        assert result.config == config
