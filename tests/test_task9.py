"""タスク9: ユニットテストと統合テストの実装

タスク9.1: DataLoaderのユニットテスト
タスク9.2: ParameterValidatorのユニットテスト
タスク9.3: NoiseFilterのユニットテスト
タスク9.4: AnomalyDetectorのユニットテスト
タスク9.5: パイプライン統合テスト
タスク9.6: CH別チャートのホバーデータ検証テスト
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from src.anomaly_detector import AnomalyDetector, AnomalyResult
from src.data_loader import REQUIRED_COLUMNS, DataLoader, LoadError
from src.noise_filter import FilterConfig, FilterResult, NoiseFilter
from src.parameter_validator import (
    InvalidWindowError,
    ParameterValidator,
)
from src.signal_analyzer import FFTResult, RMSResult, STFTResult, SignalAnalyzer
from src.visualizer import Visualizer, VisualizerConfig


# ─────────────────────────────────────────────
# テスト用フィクスチャ（ヘルパー関数）
# ─────────────────────────────────────────────


def make_full_columns_df(rows: int = 20) -> pd.DataFrame:
    """全必須列を含む小規模 DataFrame を生成するヘルパー。

    タスク 9.1 で使用する fixture DataFrame。
    """
    return pd.DataFrame(
        {
            "キロ程": [float(i) * 0.05 for i in range(rows)],
            "摩耗_測定値": [40.0 - i * 0.1 for i in range(rows)],
            "CH": ([1, 2, 3, 4] * (rows // 4 + 1))[:rows],
            "箇所名": [f"箇所{i}" for i in range(rows)],
            "通称線名名称": [f"線名{i}" for i in range(rows)],
            "駅・駅々間名称": [f"駅間{i}" for i in range(rows)],
            "電柱番号": [f"電柱{i}" for i in range(rows)],
            "架線構造名": [f"架線{i}" for i in range(rows)],
            "トロリ線種": [f"線種{i}" for i in range(rows)],
            "降雨フラグ": ["なし"] * rows,
        }
    )


def make_wear_series_for_pipeline(n: int = 50) -> pd.Series:
    """パイプライン統合テスト用の摩耗値 Series を生成するヘルパー。"""
    rng = np.random.default_rng(42)
    base = np.linspace(40.0, 38.0, n)
    noise = rng.normal(0, 0.2, n)
    return pd.Series(base + noise, name="摩耗_測定値")


def make_kilometric_series(n: int = 50) -> pd.Series:
    """キロ程 Series を生成するヘルパー。"""
    return pd.Series([float(i) * 0.05 for i in range(n)], name="キロ程")


def make_uniform_series(n: int = 20, value: float = 5.0) -> pd.Series:
    """均一値 Series を生成するヘルパー（ゼロ除算テスト用）。"""
    return pd.Series([value] * n, name="均一値")


def make_spike_series(n: int = 30, spike_idx: int = 20) -> pd.Series:
    """スパイクを含む Series を生成するヘルパー（異常点テスト用）。"""
    data = np.ones(n) * 10.0
    data[spike_idx] = 0.0  # 急激な低下 → 大きな Z-Score
    return pd.Series(data, name="スパイクデータ")


def make_channel_data_dict(n: int = 20) -> dict[int, pd.DataFrame]:
    """CH1〜4の辞書データを生成するヘルパー。"""
    result: dict[int, pd.DataFrame] = {}
    for ch in range(1, 5):
        df = make_full_columns_df(n)
        df["CH"] = ch
        result[ch] = df
    return result


# ─────────────────────────────────────────────
# タスク 9.1: DataLoaderのユニットテスト
# Requirements: 1.2, 1.3, 1.4
# ─────────────────────────────────────────────


class TestDataLoaderAllColumnsPresent:
    """9.1 必須列がすべて揃っている場合の列検証動作テスト（要件 1.2）"""

    def test_validate_columns_all_present_returns_none(self) -> None:
        """全必須列が存在する場合、validate_columns は None を返す"""
        loader = DataLoader()
        df = make_full_columns_df()
        result = loader.validate_columns(df)
        # None が返ることで列検証成功を示す
        assert result is None, "全列が存在する場合は None を返すべき"

    def test_validate_columns_with_extra_columns_returns_none(self) -> None:
        """余分な列があっても必須列が揃っていれば None を返す"""
        loader = DataLoader()
        df = make_full_columns_df()
        df["余分な列1"] = 0
        df["余分な列2"] = "テスト"
        result = loader.validate_columns(df)
        assert result is None, "余分な列があっても必須列が揃えば None を返すべき"

    def test_validate_columns_confirms_all_10_required_columns(self) -> None:
        """REQUIRED_COLUMNS に定義された10列が全て検証対象であることを確認"""
        assert len(REQUIRED_COLUMNS) == 10, "必須列は10列であるべき"
        # 10列が全て含まれているかを検証
        loader = DataLoader()
        df = make_full_columns_df()
        result = loader.validate_columns(df)
        assert result is None


class TestDataLoaderPartialColumnsMissing:
    """9.1 必須列が一部欠損の場合の列検証動作テスト（要件 1.2, 1.4）"""

    def test_one_column_missing_returns_load_error(self) -> None:
        """必須列が1列欠損の場合、LoadError を返す"""
        loader = DataLoader()
        df = make_full_columns_df().drop(columns=["キロ程"])
        result = loader.validate_columns(df)
        assert isinstance(result, LoadError), "列欠損時は LoadError を返すべき"

    def test_one_column_missing_kind_is_missing_columns(self) -> None:
        """列欠損エラーの kind は 'missing_columns' である"""
        loader = DataLoader()
        df = make_full_columns_df().drop(columns=["摩耗_測定値"])
        result = loader.validate_columns(df)
        assert isinstance(result, LoadError)
        assert result.kind == "missing_columns"

    def test_missing_column_name_in_error_list(self) -> None:
        """欠損列名がエラーの missing_columns リストに含まれる"""
        loader = DataLoader()
        dropped_col = "電柱番号"
        df = make_full_columns_df().drop(columns=[dropped_col])
        result = loader.validate_columns(df)
        assert isinstance(result, LoadError)
        assert dropped_col in result.missing_columns, (
            f"欠損列 '{dropped_col}' が missing_columns に含まれるべき"
        )

    def test_multiple_missing_columns_all_listed(self) -> None:
        """複数列欠損時、全欠損列が missing_columns に含まれる"""
        loader = DataLoader()
        missing = ["キロ程", "CH", "架線構造名"]
        df = make_full_columns_df().drop(columns=missing)
        result = loader.validate_columns(df)
        assert isinstance(result, LoadError)
        for col in missing:
            assert col in result.missing_columns, (
                f"欠損列 '{col}' が missing_columns に含まれるべき"
            )

    def test_missing_columns_message_contains_column_name(self) -> None:
        """エラーメッセージに欠損列名が含まれる（要件 1.4）"""
        loader = DataLoader()
        dropped_col = "トロリ線種"
        df = make_full_columns_df().drop(columns=[dropped_col])
        result = loader.validate_columns(df)
        assert isinstance(result, LoadError)
        assert dropped_col in result.message, (
            f"エラーメッセージに欠損列名 '{dropped_col}' が含まれるべき"
        )


class TestDataLoaderAllColumnsMissing:
    """9.1 必須列が全欠損の場合の列検証動作テスト（要件 1.2, 1.4）"""

    def test_all_columns_missing_returns_load_error(self) -> None:
        """全必須列が欠損の場合、LoadError を返す"""
        loader = DataLoader()
        # 無関係な列だけを持つ DataFrame
        df = pd.DataFrame({"無関係な列A": [1, 2, 3], "無関係な列B": ["x", "y", "z"]})
        result = loader.validate_columns(df)
        assert isinstance(result, LoadError)

    def test_all_columns_missing_kind_is_missing_columns(self) -> None:
        """全列欠損エラーの kind は 'missing_columns' である"""
        loader = DataLoader()
        df = pd.DataFrame({"無関係な列": [1, 2, 3]})
        result = loader.validate_columns(df)
        assert isinstance(result, LoadError)
        assert result.kind == "missing_columns"

    def test_all_columns_missing_count_equals_required(self) -> None:
        """全列欠損時、missing_columns の要素数が必須列数と一致する"""
        loader = DataLoader()
        df = pd.DataFrame({"無関係な列": [1, 2, 3]})
        result = loader.validate_columns(df)
        assert isinstance(result, LoadError)
        assert len(result.missing_columns) == len(REQUIRED_COLUMNS), (
            f"全必須列欠損時は missing_columns が {len(REQUIRED_COLUMNS)} 要素であるべき"
        )

    def test_empty_dataframe_all_columns_missing(self) -> None:
        """空の DataFrame（列なし）でも全欠損として処理する"""
        loader = DataLoader()
        df = pd.DataFrame()
        result = loader.validate_columns(df)
        assert isinstance(result, LoadError)
        assert result.kind == "missing_columns"
        assert len(result.missing_columns) == len(REQUIRED_COLUMNS)


class TestDataLoaderFileNotFound:
    """9.1 ファイル未存在時のエラー返却テスト（要件 1.3）"""

    def test_nonexistent_file_returns_load_error(self) -> None:
        """存在しないファイルは LoadError を返す（例外を送出しない）"""
        loader = DataLoader()
        result = loader.load("/tmp/nonexistent_task9_test_xyz.xlsx")
        assert isinstance(result, LoadError), "存在しないファイルは LoadError を返すべき"

    def test_nonexistent_file_kind_is_file_not_found(self) -> None:
        """ファイル未存在エラーの kind は 'file_not_found' である"""
        loader = DataLoader()
        result = loader.load("/tmp/nonexistent_task9_test_xyz.xlsx")
        assert isinstance(result, LoadError)
        assert result.kind == "file_not_found", (
            f"kind は 'file_not_found' であるべき。実際: {result.kind if isinstance(result, LoadError) else 'N/A'}"
        )

    def test_nonexistent_file_no_exception_raised(self) -> None:
        """ファイル未存在時に例外が外部に送出されない（要件 1.3）"""
        loader = DataLoader()
        # この呼び出しで例外が発生すると pytest がテスト失敗として記録する
        result = loader.load("/tmp/surely_nonexistent_file_42424242.xlsx")
        # 例外なく結果が返ること
        assert result is not None

    def test_nonexistent_file_message_is_nonempty(self) -> None:
        """ファイル未存在エラーメッセージが空でない"""
        loader = DataLoader()
        result = loader.load("/tmp/nonexistent_task9_test_xyz.xlsx")
        assert isinstance(result, LoadError)
        assert len(result.message) > 0, "エラーメッセージが空であってはならない"

    def test_invalid_xlsx_extension_but_not_excel(self, tmp_path: pytest.TempPathFactory) -> None:
        """xlsx 拡張子だが実際は Excel でないファイルは read_error を返す"""
        fake_file = tmp_path / "not_real.xlsx"
        fake_file.write_text("これは Excel バイナリではありません")
        loader = DataLoader()
        result = loader.load(str(fake_file))
        assert isinstance(result, LoadError)
        # file_not_found でないことを確認
        assert result.kind != "file_not_found"


# ─────────────────────────────────────────────
# タスク 9.2: ParameterValidatorのユニットテスト
# Requirements: 3.1, 3.6
# ─────────────────────────────────────────────


class TestParameterValidatorWindowValidation:
    """9.2 ウィンドウ幅バリデーション動作テスト（要件 3.1, 3.6）"""

    def test_valid_window_size_returns_int(self) -> None:
        """有効なウィンドウ幅は整数をそのまま返す"""
        validator = ParameterValidator()
        result = validator.validate_window(window_size=10, data_length=100)
        assert isinstance(result, int)
        assert result == 10

    def test_minimum_valid_window_size_1(self) -> None:
        """最小有効値 window_size=1 は通過する"""
        validator = ParameterValidator()
        result = validator.validate_window(window_size=1, data_length=100)
        assert isinstance(result, int)
        assert result == 1

    def test_maximum_valid_window_equals_data_length(self) -> None:
        """window_size == data_length は有効範囲内（最大値）"""
        validator = ParameterValidator()
        data_length = 50
        result = validator.validate_window(window_size=data_length, data_length=data_length)
        assert isinstance(result, int)
        assert result == data_length

    def test_zero_window_size_returns_invalid_window_error(self) -> None:
        """ウィンドウ幅 0 は InvalidWindowError を返す"""
        validator = ParameterValidator()
        result = validator.validate_window(window_size=0, data_length=100)
        assert isinstance(result, InvalidWindowError), "ウィンドウ幅 0 は InvalidWindowError を返すべき"

    def test_negative_window_size_returns_invalid_window_error(self) -> None:
        """負のウィンドウ幅は InvalidWindowError を返す"""
        validator = ParameterValidator()
        result = validator.validate_window(window_size=-5, data_length=100)
        assert isinstance(result, InvalidWindowError), "負のウィンドウ幅は InvalidWindowError を返すべき"

    def test_window_size_exceeds_data_length_returns_invalid_window_error(self) -> None:
        """データ長を超えるウィンドウ幅は InvalidWindowError を返す"""
        validator = ParameterValidator()
        result = validator.validate_window(window_size=101, data_length=100)
        assert isinstance(result, InvalidWindowError), "データ長超過は InvalidWindowError を返すべき"

    def test_invalid_window_error_contains_window_size(self) -> None:
        """InvalidWindowError に検証失敗した window_size が記録される"""
        validator = ParameterValidator()
        invalid_size = 0
        result = validator.validate_window(window_size=invalid_size, data_length=100)
        assert isinstance(result, InvalidWindowError)
        assert result.window_size == invalid_size

    def test_invalid_window_error_contains_data_length(self) -> None:
        """InvalidWindowError に data_length が記録される"""
        validator = ParameterValidator()
        data_len = 75
        result = validator.validate_window(window_size=200, data_length=data_len)
        assert isinstance(result, InvalidWindowError)
        assert result.data_length == data_len

    def test_invalid_window_message_contains_valid_range(self) -> None:
        """エラーメッセージに有効範囲（1 以上 data_length 以下）が含まれる"""
        validator = ParameterValidator()
        data_length = 100
        result = validator.validate_window(window_size=0, data_length=data_length)
        assert isinstance(result, InvalidWindowError)
        # メッセージに下限(1)と上限(data_length)が含まれることを確認
        assert "1" in result.message
        assert str(data_length) in result.message


class TestParameterValidatorOddWindowCorrection:
    """9.2 ウィンドウ幅奇数補正動作テスト（要件 3.6）"""

    def test_even_window_size_corrected_to_next_odd(self) -> None:
        """偶数ウィンドウ幅は +1 して奇数に補正される"""
        validator = ParameterValidator()
        result = validator.ensure_odd_window(4)
        assert result == 5

    def test_odd_window_size_unchanged(self) -> None:
        """奇数ウィンドウ幅はそのまま返される"""
        validator = ParameterValidator()
        result = validator.ensure_odd_window(5)
        assert result == 5

    def test_minimum_value_1_unchanged(self) -> None:
        """最小値 1（奇数）はそのまま返す（要件 3.6）"""
        validator = ParameterValidator()
        result = validator.ensure_odd_window(1)
        assert result == 1

    def test_even_2_becomes_3(self) -> None:
        """偶数 2 は 3 に補正される"""
        validator = ParameterValidator()
        result = validator.ensure_odd_window(2)
        assert result == 3

    def test_even_10_becomes_11(self) -> None:
        """偶数 10 は 11 に補正される"""
        validator = ParameterValidator()
        result = validator.ensure_odd_window(10)
        assert result == 11

    def test_odd_9_unchanged(self) -> None:
        """奇数 9 はそのまま 9 を返す"""
        validator = ParameterValidator()
        result = validator.ensure_odd_window(9)
        assert result == 9

    def test_result_is_always_odd_for_range_1_to_20(self) -> None:
        """1〜20 の全入力値に対して結果が常に奇数である（要件 3.6）"""
        validator = ParameterValidator()
        for w in range(1, 21):
            result = validator.ensure_odd_window(w)
            assert result % 2 == 1, (
                f"window={w} の補正結果 {result} が奇数でない"
            )


# ─────────────────────────────────────────────
# タスク 9.3: NoiseFilterのユニットテスト
# Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6
# ─────────────────────────────────────────────


class TestNoiseFilterBothOff:
    """9.3 両方 OFF の場合のテスト（要件 4.5, 4.6）"""

    def test_both_off_returns_filter_result(self) -> None:
        """両フィルタ OFF で FilterResult が返される"""
        nf = NoiseFilter()
        series = make_wear_series_for_pipeline()
        config = FilterConfig(median_enabled=False, savgol_enabled=False)
        result = nf.apply(series, config)
        assert isinstance(result, FilterResult)

    def test_both_off_original_equals_filtered(self) -> None:
        """両フィルタ OFF で original と filtered が同一値（要件 4.6）"""
        nf = NoiseFilter()
        series = make_wear_series_for_pipeline()
        config = FilterConfig(median_enabled=False, savgol_enabled=False)
        result = nf.apply(series, config)
        # 両フィルタ OFF では original == filtered であるべき
        pd.testing.assert_series_equal(
            result.original,
            result.filtered,
            check_names=False,
        )

    def test_both_off_original_stored_correctly(self) -> None:
        """両フィルタ OFF で元データが result.original に保持される"""
        nf = NoiseFilter()
        series = make_wear_series_for_pipeline(n=30)
        config = FilterConfig(median_enabled=False, savgol_enabled=False)
        result = nf.apply(series, config)
        pd.testing.assert_series_equal(result.original, series, check_names=False)

    def test_both_off_config_stored(self) -> None:
        """使用した FilterConfig が result.config に保持される"""
        nf = NoiseFilter()
        series = make_wear_series_for_pipeline()
        config = FilterConfig(median_enabled=False, savgol_enabled=False)
        result = nf.apply(series, config)
        assert result.config == config


class TestNoiseFilterMedianOnly:
    """9.3 移動中央値のみ ON の場合のテスト（要件 4.1, 4.3）"""

    def test_median_only_returns_filter_result(self) -> None:
        """移動中央値のみ ON で FilterResult が返される"""
        nf = NoiseFilter()
        series = make_wear_series_for_pipeline()
        config = FilterConfig(median_enabled=True, savgol_enabled=False, median_window=5)
        result = nf.apply(series, config)
        assert isinstance(result, FilterResult)

    def test_median_only_filtered_differs_from_original(self) -> None:
        """移動中央値適用後のデータはノイズデータと異なる（フィルタ効果の確認）"""
        nf = NoiseFilter()
        # ノイズを含む Series（移動中央値で変化する）
        rng = np.random.default_rng(0)
        noisy = pd.Series(np.linspace(10.0, 9.0, 100) + rng.normal(0, 0.5, 100))
        config = FilterConfig(median_enabled=True, savgol_enabled=False, median_window=7)
        result = nf.apply(noisy, config)
        # 完全に同一値になることはない（ノイズが平滑化される）
        assert not result.original.equals(result.filtered), (
            "移動中央値フィルタ後のデータは元ノイズデータと異なるべき"
        )

    def test_median_only_no_nan_in_filtered(self) -> None:
        """移動中央値フィルタ後のデータに NaN が含まれない（min_periods=1 の効果）"""
        nf = NoiseFilter()
        series = make_wear_series_for_pipeline(n=30)
        config = FilterConfig(median_enabled=True, savgol_enabled=False, median_window=5)
        result = nf.apply(series, config)
        assert result.filtered.isna().sum() == 0, "移動中央値フィルタ後に NaN が含まれてはならない"

    def test_median_only_original_preserved(self) -> None:
        """移動中央値のみ ON で result.original に元データが保持される"""
        nf = NoiseFilter()
        series = make_wear_series_for_pipeline()
        config = FilterConfig(median_enabled=True, savgol_enabled=False)
        result = nf.apply(series, config)
        pd.testing.assert_series_equal(result.original, series, check_names=False)


class TestNoiseFilterSavgolOnly:
    """9.3 Savitzky-Golay のみ ON の場合のテスト（要件 4.2, 4.4）"""

    def test_savgol_only_returns_filter_result(self) -> None:
        """Savitzky-Golay のみ ON で FilterResult が返される"""
        nf = NoiseFilter()
        series = make_wear_series_for_pipeline(n=50)
        config = FilterConfig(
            median_enabled=False, savgol_enabled=True, savgol_window=11, savgol_polyorder=2
        )
        result = nf.apply(series, config)
        assert isinstance(result, FilterResult)

    def test_savgol_only_filtered_length_unchanged(self) -> None:
        """Savitzky-Golay フィルタ後のデータ長が変わらない"""
        nf = NoiseFilter()
        series = make_wear_series_for_pipeline(n=50)
        config = FilterConfig(median_enabled=False, savgol_enabled=True, savgol_window=11)
        result = nf.apply(series, config)
        assert len(result.filtered) == len(series)

    def test_savgol_only_original_preserved(self) -> None:
        """Savitzky-Golay のみ ON で result.original に元データが保持される"""
        nf = NoiseFilter()
        series = make_wear_series_for_pipeline(n=50)
        config = FilterConfig(median_enabled=False, savgol_enabled=True, savgol_window=11)
        result = nf.apply(series, config)
        pd.testing.assert_series_equal(result.original, series, check_names=False)

    def test_savgol_only_smoothing_effect(self) -> None:
        """Savitzky-Golay フィルタが平滑化効果を持つことを確認する"""
        nf = NoiseFilter()
        rng = np.random.default_rng(1)
        series = pd.Series(np.linspace(10.0, 9.0, 100) + rng.normal(0, 0.5, 100))
        config = FilterConfig(median_enabled=False, savgol_enabled=True, savgol_window=11)
        result = nf.apply(series, config)
        # フィルタ後の標準偏差がフィルタ前より小さい（ノイズ除去の効果）
        assert result.filtered.std() < result.original.std(), (
            "Savitzky-Golay フィルタ後の分散が元データより小さいべき"
        )


class TestNoiseFilterBothOn:
    """9.3 両方 ON の場合のテスト（移動中央値 → SG の順、要件 4.5）"""

    def test_both_on_returns_filter_result(self) -> None:
        """両フィルタ ON で FilterResult が返される"""
        nf = NoiseFilter()
        series = make_wear_series_for_pipeline(n=100)
        config = FilterConfig(
            median_enabled=True,
            savgol_enabled=True,
            median_window=5,
            savgol_window=11,
            savgol_polyorder=2,
        )
        result = nf.apply(series, config)
        assert isinstance(result, FilterResult)

    def test_both_on_applies_median_then_savgol_order(self) -> None:
        """両方 ON では移動中央値 → SG の順で適用される（要件 4.5）

        検証方法: 両方 ON の結果が、手動で中央値 → SG の順に適用した結果と一致する。
        """
        nf = NoiseFilter()
        series = make_wear_series_for_pipeline(n=100)
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
        result_both = nf.apply(series, config_both)

        # 手動で順序通りに適用した場合（中央値 → SG）
        after_median = nf.apply_rolling_median(series, window=median_w)
        expected_filtered = nf.apply_savgol(after_median, window=savgol_w, polyorder=polyorder)

        # 両フィルタ ON は 中央値 → SG の順で適用されるべき
        pd.testing.assert_series_equal(
            result_both.filtered,
            expected_filtered,
            check_names=False,
        )

    def test_both_on_original_preserved(self) -> None:
        """両フィルタ ON で元データが result.original に保持される"""
        nf = NoiseFilter()
        series = make_wear_series_for_pipeline(n=100)
        config = FilterConfig(median_enabled=True, savgol_enabled=True)
        result = nf.apply(series, config)
        pd.testing.assert_series_equal(result.original, series, check_names=False)

    def test_both_on_filtered_length_unchanged(self) -> None:
        """両フィルタ適用後もデータ長が変わらない"""
        nf = NoiseFilter()
        series = make_wear_series_for_pipeline(n=100)
        config = FilterConfig(median_enabled=True, savgol_enabled=True)
        result = nf.apply(series, config)
        assert len(result.filtered) == len(series)


# ─────────────────────────────────────────────
# タスク 9.4: AnomalyDetectorのユニットテスト
# Requirements: 5.1, 5.2, 5.4, 5.6
# ─────────────────────────────────────────────


class TestAnomalyDetectorWindowUnderfillNaN:
    """9.4 ウィンドウ未満区間が NaN になることを検証（要件 5.1, 5.6）"""

    def test_moving_average_nan_for_underfilled_window(self) -> None:
        """compute_moving_average: ウィンドウが満たない先頭区間は NaN（要件 5.1）"""
        detector = AnomalyDetector()
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        window_size = 3
        result = detector.compute_moving_average(series, window_size=window_size)
        # 先頭 window_size - 1 = 2 点は NaN
        assert pd.isna(result.iloc[0]), "ウィンドウ未満の先頭点は NaN であるべき"
        assert pd.isna(result.iloc[1]), "ウィンドウ未満の先頭点は NaN であるべき"
        # ウィンドウが満たされた点は有効値
        assert not pd.isna(result.iloc[2])

    def test_zscore_nan_for_underfilled_window(self) -> None:
        """compute_zscore: ウィンドウが満たない先頭区間は NaN（要件 5.6）"""
        detector = AnomalyDetector()
        series = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        window_size = 3
        result = detector.compute_zscore(series, window_size=window_size)
        # 先頭 window_size - 1 点は NaN
        assert pd.isna(result.iloc[0]), "ウィンドウ未満の Z-Score は NaN であるべき"
        assert pd.isna(result.iloc[1]), "ウィンドウ未満の Z-Score は NaN であるべき"

    def test_zscore_series_length_equals_input_length(self) -> None:
        """Z-Score Series の長さは入力 Series と等しい（不変条件）"""
        detector = AnomalyDetector()
        n = 50
        series = make_wear_series_for_pipeline(n=n)
        result = detector.compute_zscore(series, window_size=5)
        assert len(result) == n, "Z-Score Series の長さは入力長と同じであるべき"

    def test_detect_zscore_length_equals_input(self) -> None:
        """detect() の zscore_series 長さは入力 Series と等しい"""
        detector = AnomalyDetector()
        n = 30
        series = make_spike_series(n=n)
        result = detector.detect(series, window_size=5, threshold=2.0)
        assert len(result.zscore_series) == n


class TestAnomalyDetectorZeroDivisionNaN:
    """9.4 ゼロ除算が NaN になることを検証（要件 5.2, 5.6）"""

    def test_uniform_series_zscore_is_nan(self) -> None:
        """均一な値の Series（標準偏差=0）では Z-Score が NaN になる（ゼロ除算ガード）"""
        detector = AnomalyDetector()
        # 全て同一値 → 標準偏差=0 → ゼロ除算
        series = make_uniform_series(n=20, value=5.0)
        result = detector.compute_zscore(series, window_size=3)
        # ウィンドウが満たされた区間（インデックス2以降）の Z-Score が NaN
        valid_region = result.iloc[3:]  # ウィンドウを確実に満たした点
        assert valid_region.isna().all(), (
            "均一な値（std=0）の Z-Score はゼロ除算を防ぐため NaN であるべき"
        )

    def test_zero_std_does_not_raise_exception(self) -> None:
        """標準偏差がゼロの区間で例外が発生しない（ゼロ除算ガード）"""
        detector = AnomalyDetector()
        series = make_uniform_series(n=15, value=3.0)
        # 例外が発生すると pytest がテスト失敗として記録する
        result = detector.compute_zscore(series, window_size=3)
        assert result is not None

    def test_detect_with_uniform_series_no_anomalies(self) -> None:
        """均一データでの異常検知: Z-Score が NaN なので閾値超過なし"""
        detector = AnomalyDetector()
        series = make_uniform_series(n=20, value=5.0)
        result = detector.detect(series, window_size=3, threshold=2.0)
        # NaN は閾値超過として扱われないため anomaly_indices は空
        assert isinstance(result, AnomalyResult)
        assert len(result.anomaly_indices) == 0, (
            "均一データ（std=0）での異常点は NaN として閾値比較対象外になるべき"
        )


class TestAnomalyDetectorThresholdRecording:
    """9.4 閾値超過点が正しく記録されることを検証（要件 5.4）"""

    def test_spike_detected_as_anomaly(self) -> None:
        """スパイク点が異常として検知され anomaly_indices に記録される"""
        detector = AnomalyDetector()
        spike_idx = 20
        series = make_spike_series(n=30, spike_idx=spike_idx)
        # ウィンドウ幅5でスパイクの Z-Score は大きい
        result = detector.detect(series, window_size=5, threshold=1.5)
        assert spike_idx in result.anomaly_indices, (
            f"スパイク点（インデックス {spike_idx}）が anomaly_indices に含まれるべき"
        )

    def test_anomaly_indices_subset_of_zscore_index(self) -> None:
        """anomaly_indices は zscore_series のインデックスのサブセットである（不変条件）"""
        detector = AnomalyDetector()
        series = make_spike_series(n=30, spike_idx=20)
        result = detector.detect(series, window_size=5, threshold=1.5)
        for idx in result.anomaly_indices:
            assert idx in result.zscore_series.index, (
                f"anomaly_index {idx} が zscore_series.index に含まれるべき"
            )

    def test_all_anomaly_indices_exceed_threshold(self) -> None:
        """anomaly_indices の全点で |Z-Score| >= threshold である（要件 5.4）"""
        detector = AnomalyDetector()
        series = make_spike_series(n=30, spike_idx=20)
        threshold = 1.5
        result = detector.detect(series, window_size=5, threshold=threshold)
        assert len(result.anomaly_indices) > 0, "スパイクが検知されなかった"
        for idx in result.anomaly_indices:
            zscore_val = result.zscore_series.iloc[idx]
            assert abs(zscore_val) >= threshold, (
                f"anomaly_index {idx} の |Z-Score| ({abs(zscore_val):.4f}) が"
                f" threshold ({threshold}) 以上であるべき"
            )

    def test_very_high_threshold_no_anomaly(self) -> None:
        """非常に高い閾値では異常点が記録されない"""
        detector = AnomalyDetector()
        series = make_spike_series(n=30, spike_idx=20)
        result = detector.detect(series, window_size=5, threshold=1000.0)
        assert len(result.anomaly_indices) == 0

    def test_threshold_stored_in_result(self) -> None:
        """指定した閾値が AnomalyResult.threshold に記録される"""
        detector = AnomalyDetector()
        series = make_spike_series(n=30)
        threshold = 3.14159
        result = detector.detect(series, window_size=5, threshold=threshold)
        assert result.threshold == pytest.approx(threshold)

    def test_anomaly_positions_recorded_with_position_series(self) -> None:
        """position_series を渡した場合、異常点のキロ程値が anomaly_positions に記録される"""
        detector = AnomalyDetector()
        spike_idx = 20
        n = 30
        series = make_spike_series(n=n, spike_idx=spike_idx)
        positions = make_kilometric_series(n=n)
        result = detector.detect(series, window_size=5, threshold=1.5, position_series=positions)
        # スパイク点が検知されること
        assert spike_idx in result.anomaly_indices
        # キロ程値が記録されていること
        assert len(result.anomaly_positions) > 0
        # 最初の異常点のキロ程値を確認
        expected_position = float(spike_idx) * 0.05
        # anomaly_positions にスパイク点のキロ程が含まれる
        assert expected_position in result.anomaly_positions.values or pytest.approx(
            expected_position
        ) == result.anomaly_positions.iloc[0]


# ─────────────────────────────────────────────
# タスク 9.5: パイプライン統合テスト
# Requirements: 3.2, 3.3, 3.4, 5.1, 5.2, 5.4, 6.1
# ─────────────────────────────────────────────


class TestPipelineIntegrationDataLoaderToAnomalyDetector:
    """9.5 DataLoader → NoiseFilter → AnomalyDetector パイプライン統合テスト（要件 5.1, 5.2, 5.4）"""

    def _make_fixture_df(self, n: int = 50) -> pd.DataFrame:
        """小規模 fixture DataFrame を生成するヘルパー。"""
        return make_full_columns_df(rows=n)

    def test_pipeline_produces_anomaly_result(self) -> None:
        """フルパイプラインが AnomalyResult を返す（型検証）"""
        # Step1: DataLoader でデータを取得（fixture DataFrame を使用）
        df = self._make_fixture_df(n=50)

        # Step2: NoiseFilter でフィルタリング
        nf = NoiseFilter()
        config = FilterConfig(
            median_enabled=True, median_window=5, savgol_enabled=False
        )
        wear_series = df["摩耗_測定値"]
        filter_result = nf.apply(wear_series, config)

        # Step3: AnomalyDetector で異常検知
        detector = AnomalyDetector()
        anomaly_result = detector.detect(
            filter_result.filtered,
            window_size=5,
            threshold=2.0,
            position_series=df["キロ程"],
        )

        # 型検証
        assert isinstance(anomaly_result, AnomalyResult)

    def test_pipeline_anomaly_result_invariants(self) -> None:
        """AnomalyResult の不変条件検証（要件 5.4）

        - zscore_series の長さは入力と等しい
        - anomaly_indices は zscore_series のサブセット
        - threshold が正しく記録されている
        """
        df = self._make_fixture_df(n=50)
        nf = NoiseFilter()
        config = FilterConfig(median_enabled=True, median_window=5, savgol_enabled=False)
        wear_series = df["摩耗_測定値"]
        filter_result = nf.apply(wear_series, config)

        detector = AnomalyDetector()
        threshold = 2.0
        anomaly_result = detector.detect(
            filter_result.filtered,
            window_size=5,
            threshold=threshold,
            position_series=df["キロ程"],
        )

        # 不変条件: zscore_series 長さ == 入力長
        assert len(anomaly_result.zscore_series) == len(filter_result.filtered), (
            "zscore_series の長さは入力 Series と等しいべき"
        )

        # 不変条件: anomaly_indices は zscore_series のサブセット
        for idx in anomaly_result.anomaly_indices:
            assert idx in anomaly_result.zscore_series.index

        # 不変条件: threshold が正しく保存される
        assert anomaly_result.threshold == pytest.approx(threshold)

    def test_pipeline_anomaly_result_contains_moving_average(self) -> None:
        """AnomalyResult に moving_average が含まれる（要件 5.1）"""
        df = self._make_fixture_df(n=50)
        wear_series = df["摩耗_測定値"]

        detector = AnomalyDetector()
        anomaly_result = detector.detect(wear_series, window_size=5, threshold=2.0)

        assert isinstance(anomaly_result.moving_average, pd.Series)
        assert len(anomaly_result.moving_average) == len(wear_series)

    def test_pipeline_with_both_filters_enabled(self) -> None:
        """両フィルタ ON → 異常検知のパイプラインが正常動作する"""
        df = self._make_fixture_df(n=100)
        wear_series = df["摩耗_測定値"]

        nf = NoiseFilter()
        config = FilterConfig(
            median_enabled=True, median_window=5,
            savgol_enabled=True, savgol_window=11, savgol_polyorder=2
        )
        filter_result = nf.apply(wear_series, config)

        detector = AnomalyDetector()
        anomaly_result = detector.detect(
            filter_result.filtered, window_size=10, threshold=2.0
        )

        assert isinstance(anomaly_result, AnomalyResult)
        assert len(anomaly_result.zscore_series) == len(wear_series)


class TestSignalAnalyzerFFTSTFTDimensions:
    """9.5 SignalAnalyzer の FFT・STFT 出力次元がウィンドウ幅に基づく期待値と一致する（要件 3.3, 3.4）"""

    def test_fft_frequencies_length_is_window_size_half_plus_1(self) -> None:
        """FFT の frequencies 長さは window_size // 2 + 1（rfft の仕様）"""
        analyzer = SignalAnalyzer()
        series = make_wear_series_for_pipeline(n=200)
        window_size = 64
        result = analyzer.compute_fft(series, window_size=window_size)
        expected_len = window_size // 2 + 1
        assert len(result.frequencies) == expected_len, (
            f"FFT frequencies 長さは {expected_len} であるべき（実際: {len(result.frequencies)}）"
        )

    def test_fft_amplitudes_length_matches_frequencies(self) -> None:
        """FFT の amplitudes 長さは frequencies と等しい"""
        analyzer = SignalAnalyzer()
        series = make_wear_series_for_pipeline(n=200)
        result = analyzer.compute_fft(series, window_size=32)
        assert len(result.amplitudes) == len(result.frequencies)

    def test_fft_frequencies_length_varies_with_window_size(self) -> None:
        """window_size が異なれば FFT 出力長も異なる"""
        analyzer = SignalAnalyzer()
        series = make_wear_series_for_pipeline(n=200)
        result_32 = analyzer.compute_fft(series, window_size=32)
        result_64 = analyzer.compute_fft(series, window_size=64)
        assert len(result_32.frequencies) == 32 // 2 + 1
        assert len(result_64.frequencies) == 64 // 2 + 1
        assert len(result_32.frequencies) != len(result_64.frequencies)

    def test_stft_frequencies_length_is_window_size_half_plus_1(self) -> None:
        """STFT の frequencies 長さは window_size // 2 + 1"""
        analyzer = SignalAnalyzer()
        series = make_wear_series_for_pipeline(n=200)
        window_size = 32
        result = analyzer.compute_stft(series, window_size=window_size)
        expected_len = window_size // 2 + 1
        assert len(result.frequencies) == expected_len, (
            f"STFT frequencies 長さは {expected_len} であるべき（実際: {len(result.frequencies)}）"
        )

    def test_stft_spectrogram_shape_is_freq_times_positions(self) -> None:
        """STFT の spectrogram 形状は [len(frequencies), len(positions)]"""
        analyzer = SignalAnalyzer()
        series = make_wear_series_for_pipeline(n=200)
        result = analyzer.compute_stft(series, window_size=32)
        expected_shape = (len(result.frequencies), len(result.positions))
        assert result.spectrogram.shape == expected_shape, (
            f"spectrogram 形状は {expected_shape} であるべき（実際: {result.spectrogram.shape}）"
        )

    def test_fft_result_type(self) -> None:
        """compute_fft の戻り値型は FFTResult である"""
        analyzer = SignalAnalyzer()
        series = make_wear_series_for_pipeline(n=100)
        result = analyzer.compute_fft(series, window_size=32)
        assert isinstance(result, FFTResult)

    def test_stft_result_type(self) -> None:
        """compute_stft の戻り値型は STFTResult である"""
        analyzer = SignalAnalyzer()
        series = make_wear_series_for_pipeline(n=200)
        result = analyzer.compute_stft(series, window_size=16)
        assert isinstance(result, STFTResult)


class TestVisualizerComparisonViewSubplots:
    """9.5 Visualizer.plot_comparison_view が 3 サブプロットを持つ Figure を返す（要件 6.1）"""

    @pytest.fixture()
    def fixture_data(self):
        """統合テスト用のフィクスチャデータを準備する。"""
        n = 50
        series = make_wear_series_for_pipeline(n=n)
        kilometric = make_kilometric_series(n=n)

        # AnomalyResult を生成
        detector = AnomalyDetector()
        anomaly_result = detector.detect(series, window_size=5, threshold=2.0)

        return series, kilometric, anomaly_result

    def test_plot_comparison_view_returns_go_figure(self, fixture_data) -> None:
        """plot_comparison_view は go.Figure を返す"""
        series, kilometric, anomaly_result = fixture_data
        visualizer = Visualizer()
        config = VisualizerConfig()

        fig = visualizer.plot_comparison_view(
            raw_series=series,
            filtered_series=series,
            anomaly_result=anomaly_result,
            kilometric_series=kilometric,
            config=config,
        )
        assert isinstance(fig, go.Figure)

    def test_plot_comparison_view_has_3_main_traces(self, fixture_data) -> None:
        """統合比較ビューは生波形・フィルタ後・Z-Score の3系列プロットを持つ（要件 6.1）"""
        series, kilometric, anomaly_result = fixture_data
        visualizer = Visualizer()
        config = VisualizerConfig()

        fig = visualizer.plot_comparison_view(
            raw_series=series,
            filtered_series=series,
            anomaly_result=anomaly_result,
            kilometric_series=kilometric,
            config=config,
        )
        # 生波形 + フィルタ後 + Z-Score = 少なくとも3トレース
        # （+ 閾値ラインで5トレース以上の場合もある）
        assert len(fig.data) >= 3, (
            f"比較ビューには少なくとも3トレースが必要（実際: {len(fig.data)}）"
        )

    def test_plot_comparison_view_has_3_subplot_rows(self, fixture_data) -> None:
        """統合比較ビューは make_subplots(rows=3) で3段構成になる（要件 6.1）

        xaxis3 が存在することで3段サブプロットを確認する。
        """
        series, kilometric, anomaly_result = fixture_data
        visualizer = Visualizer()
        config = VisualizerConfig()

        fig = visualizer.plot_comparison_view(
            raw_series=series,
            filtered_series=series,
            anomaly_result=anomaly_result,
            kilometric_series=kilometric,
            config=config,
        )
        # 3段サブプロットには xaxis, xaxis2, xaxis3 が存在するはず
        layout_dict = fig.layout.to_plotly_json()
        assert "xaxis3" in layout_dict, (
            "3段サブプロットには xaxis3 が存在するべき（make_subplots rows=3 の証拠）"
        )

    def test_plot_comparison_view_shared_x_axis(self, fixture_data) -> None:
        """X軸が連動している（shared_xaxes=True、要件 6.2）"""
        series, kilometric, anomaly_result = fixture_data
        visualizer = Visualizer()
        config = VisualizerConfig()

        fig = visualizer.plot_comparison_view(
            raw_series=series,
            filtered_series=series,
            anomaly_result=anomaly_result,
            kilometric_series=kilometric,
            config=config,
        )
        layout_dict = fig.layout.to_plotly_json()
        # xaxis2 と xaxis3 が xaxis に連動している（matches="x"）
        if "xaxis2" in layout_dict:
            xaxis2 = layout_dict["xaxis2"]
            assert "matches" in xaxis2, "xaxis2 は xaxis に matches しているべき"
        # または少なくとも複数のトレースが存在することで X 軸共有を間接確認
        assert len(fig.data) >= 3

    def test_plot_comparison_view_has_threshold_line(self, fixture_data) -> None:
        """Z-Score プロットに閾値ラインが破線で含まれる（要件 6.3）"""
        series, kilometric, anomaly_result = fixture_data
        visualizer = Visualizer()
        config = VisualizerConfig()

        fig = visualizer.plot_comparison_view(
            raw_series=series,
            filtered_series=series,
            anomaly_result=anomaly_result,
            kilometric_series=kilometric,
            config=config,
        )
        # 破線（dash="dash"）のトレースが存在することを確認
        has_threshold_line = any(
            hasattr(trace, "line")
            and trace.line is not None
            and hasattr(trace.line, "dash")
            and trace.line.dash == "dash"
            for trace in fig.data
        )
        assert has_threshold_line, "閾値ラインが破線で表示されているべき（要件 6.3）"


# ─────────────────────────────────────────────
# タスク 9.6: CH別チャートのホバーデータ検証テスト
# Requirements: 2.3
# ─────────────────────────────────────────────


class TestCHChartHoverTemplateValidation:
    """9.6 CH別チャートの hovertemplate に8項目が含まれることを fig.data を解析して検証

    要件 2.3: hovertemplate に以下8項目が含まれること
      - 箇所名
      - 通称線名名称
      - 駅・駅々間名称
      - 電柱番号
      - キロ程
      - 架線構造名
      - トロリ線種
      - 降雨フラグ
    """

    # 検証対象の8項目
    REQUIRED_HOVER_ITEMS = [
        "箇所名",
        "通称線名名称",
        "駅・駅々間名称",
        "電柱番号",
        "キロ程",
        "架線構造名",
        "トロリ線種",
        "降雨フラグ",
    ]

    @pytest.fixture()
    def ch_figure(self) -> go.Figure:
        """CH別チャートの Figure を生成するフィクスチャ。"""
        visualizer = Visualizer()
        config = VisualizerConfig()
        channel_data = make_channel_data_dict(n=20)
        return visualizer.plot_channels(channel_data, config)

    def test_ch_chart_figure_has_data(self, ch_figure: go.Figure) -> None:
        """CH別チャートに fig.data が存在する（前提条件）"""
        assert len(ch_figure.data) > 0, "CH別チャートには fig.data が存在するべき"

    def test_ch1_trace_hovertemplate_contains_all_8_items(self, ch_figure: go.Figure) -> None:
        """CH1 トレースの hovertemplate に8項目が全て含まれる（要件 2.3）"""
        # 最初のトレース（CH1）を検証
        trace = ch_figure.data[0]
        hovertemplate = trace.hovertemplate
        assert hovertemplate is not None, "hovertemplate が設定されているべき"

        for item in self.REQUIRED_HOVER_ITEMS:
            assert item in hovertemplate, (
                f"hovertemplate に '{item}' が含まれるべき。"
                f"実際の hovertemplate: {hovertemplate}"
            )

    def test_all_ch_traces_have_hovertemplate(self, ch_figure: go.Figure) -> None:
        """全CH（1〜4）のトレースに hovertemplate が設定されている"""
        for i, trace in enumerate(ch_figure.data):
            assert hasattr(trace, "hovertemplate"), (
                f"トレース[{i}] に hovertemplate 属性がない"
            )
            assert trace.hovertemplate is not None, (
                f"トレース[{i}] の hovertemplate が None である"
            )

    def test_all_ch_traces_contain_all_8_hover_items(self, ch_figure: go.Figure) -> None:
        """全CH（1〜4）の各トレースの hovertemplate に8項目が含まれる（要件 2.3）"""
        for i, trace in enumerate(ch_figure.data):
            hovertemplate = trace.hovertemplate
            if hovertemplate is None:
                continue
            for item in self.REQUIRED_HOVER_ITEMS:
                assert item in hovertemplate, (
                    f"トレース[{i}]（{getattr(trace, 'name', '不明')}）の"
                    f" hovertemplate に '{item}' が含まれるべき"
                )

    def test_hovertemplate_contains_kisomei(self, ch_figure: go.Figure) -> None:
        """hovertemplate に '箇所名' が含まれる（要件 2.3 項目1）"""
        trace = ch_figure.data[0]
        assert "箇所名" in trace.hovertemplate

    def test_hovertemplate_contains_line_name(self, ch_figure: go.Figure) -> None:
        """hovertemplate に '通称線名名称' が含まれる（要件 2.3 項目2）"""
        trace = ch_figure.data[0]
        assert "通称線名名称" in trace.hovertemplate

    def test_hovertemplate_contains_station_name(self, ch_figure: go.Figure) -> None:
        """hovertemplate に '駅・駅々間名称' が含まれる（要件 2.3 項目3）"""
        trace = ch_figure.data[0]
        assert "駅・駅々間名称" in trace.hovertemplate

    def test_hovertemplate_contains_pole_number(self, ch_figure: go.Figure) -> None:
        """hovertemplate に '電柱番号' が含まれる（要件 2.3 項目4）"""
        trace = ch_figure.data[0]
        assert "電柱番号" in trace.hovertemplate

    def test_hovertemplate_contains_kilometric(self, ch_figure: go.Figure) -> None:
        """hovertemplate に 'キロ程' が含まれる（要件 2.3 項目5）"""
        trace = ch_figure.data[0]
        assert "キロ程" in trace.hovertemplate

    def test_hovertemplate_contains_overhead_structure(self, ch_figure: go.Figure) -> None:
        """hovertemplate に '架線構造名' が含まれる（要件 2.3 項目6）"""
        trace = ch_figure.data[0]
        assert "架線構造名" in trace.hovertemplate

    def test_hovertemplate_contains_trolley_wire_type(self, ch_figure: go.Figure) -> None:
        """hovertemplate に 'トロリ線種' が含まれる（要件 2.3 項目7）"""
        trace = ch_figure.data[0]
        assert "トロリ線種" in trace.hovertemplate

    def test_hovertemplate_contains_rain_flag(self, ch_figure: go.Figure) -> None:
        """hovertemplate に '降雨フラグ' が含まれる（要件 2.3 項目8）"""
        trace = ch_figure.data[0]
        assert "降雨フラグ" in trace.hovertemplate

    def test_build_hover_template_contains_all_8_items(self) -> None:
        """build_hover_template が8項目全てを含む文字列を返す（要件 2.3）"""
        visualizer = Visualizer()
        df = make_full_columns_df(rows=5)
        template = visualizer.build_hover_template(df)

        for item in self.REQUIRED_HOVER_ITEMS:
            assert item in template, (
                f"build_hover_template の返却値に '{item}' が含まれるべき"
            )

    def test_single_ch_chart_hovertemplate_has_8_items(self) -> None:
        """単一 CH のチャートでも hovertemplate に8項目が含まれる（要件 2.3）"""
        visualizer = Visualizer()
        config = VisualizerConfig()
        df = make_full_columns_df(rows=10)
        channel_data = {1: df}

        fig = visualizer.plot_channels(channel_data, config)
        assert len(fig.data) > 0

        hovertemplate = fig.data[0].hovertemplate
        assert hovertemplate is not None

        for item in self.REQUIRED_HOVER_ITEMS:
            assert item in hovertemplate, (
                f"単一CH チャートの hovertemplate に '{item}' が含まれるべき"
            )
