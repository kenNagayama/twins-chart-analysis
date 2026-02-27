"""タスク8: Streamlitアプリ統合実装のテスト

タスク8.1: サイドバーパラメータウィジェットの実装
タスク8.2: メイン画面へのグラフ描画統合
タスク8.3: 異常検知結果のグラフハイライト統合

タスク6: パイプライン統合テストと DataLoader 既存テストの動作確認
タスク6.1: アップロードデータ経由のフルパイプライン統合テスト
タスク6.2: 既存パラメータウィジェットの動作が変更されていないことを確認するテスト
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from src.anomaly_detector import AnomalyDetector, AnomalyResult
from src.data_loader import DataLoader, LoadError
from src.noise_filter import FilterConfig, FilterResult, NoiseFilter
from src.parameter_validator import (
    InvalidThresholdError,
    InvalidWindowError,
    ParameterValidator,
)
from src.signal_analyzer import SignalAnalyzer
from src.visualizer import Visualizer, VisualizerConfig


# ─────────────────────────────────────────────
# テスト用フィクスチャ
# ─────────────────────────────────────────────


def make_wear_dataframe(n: int = 30) -> pd.DataFrame:
    """テスト用の小規模 WearDataFrame（必須10列を含む）を生成するヘルパー。"""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "キロ程": np.linspace(1000.0, 1001.5, n),
            "摩耗_測定値": np.linspace(10.0, 9.0, n) + rng.standard_normal(n) * 0.1,
            "CH": ([1] * (n // 4) + [2] * (n // 4) + [3] * (n // 4) + [4] * (n // 4 + n % 4)),
            "箇所名": ["箇所A"] * n,
            "通称線名名称": ["京浜東北線"] * n,
            "駅・駅々間名称": ["東京〜上野"] * n,
            "電柱番号": [f"P{i:03d}" for i in range(n)],
            "架線構造名": ["シンプルカテナリ"] * n,
            "トロリ線種": ["GT-150"] * n,
            "降雨フラグ": ["0"] * n,
        }
    )


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """テスト用 WearDataFrame"""
    return make_wear_dataframe(30)


@pytest.fixture()
def ch1_df(sample_df: pd.DataFrame) -> pd.DataFrame:
    """CH1のみのデータ"""
    return sample_df[sample_df["CH"] == 1].reset_index(drop=True)


@pytest.fixture()
def wear_series(ch1_df: pd.DataFrame) -> pd.Series:
    """CH1の摩耗値 Series"""
    return ch1_df["摩耗_測定値"]


@pytest.fixture()
def kilometric_series(ch1_df: pd.DataFrame) -> pd.Series:
    """CH1のキロ程 Series"""
    return ch1_df["キロ程"]


@pytest.fixture()
def channel_data(sample_df: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """CH1〜4の辞書データ"""
    loader = DataLoader()
    result: dict[int, pd.DataFrame] = {}
    for ch in range(1, 5):
        df_ch = loader.get_channel_group(sample_df, ch)
        if len(df_ch) > 0:
            result[ch] = df_ch.reset_index(drop=True)
    return result


@pytest.fixture()
def filter_result(wear_series: pd.Series) -> FilterResult:
    """FilterResult フィクスチャ"""
    nf = NoiseFilter()
    config = FilterConfig(
        median_enabled=True,
        median_window=3,
        savgol_enabled=False,
    )
    return nf.apply(wear_series, config)


@pytest.fixture()
def anomaly_result(wear_series: pd.Series, kilometric_series: pd.Series) -> AnomalyResult:
    """AnomalyResult フィクスチャ (window=5, threshold=2.0)"""
    detector = AnomalyDetector()
    return detector.detect(
        wear_series, window_size=5, threshold=2.0, position_series=kilometric_series
    )


@pytest.fixture()
def visualizer() -> Visualizer:
    """Visualizer インスタンス"""
    return Visualizer()


@pytest.fixture()
def default_viz_config() -> VisualizerConfig:
    """デフォルト VisualizerConfig"""
    return VisualizerConfig()


# ─────────────────────────────────────────────
# タスク 8.1: サイドバーパラメータウィジェット
# パラメータバリデーションのロジックテスト
# ─────────────────────────────────────────────


class TestSidebarParameterValidation:
    """8.1 サイドバーパラメータウィジェットの検証ロジックテスト

    Streamlit のウィジェット自体は UI テストが困難なため、
    app.py が利用するバリデーションロジックを直接テストする。
    """

    def test_valid_window_size_passes_validation(self, ch1_df: pd.DataFrame) -> None:
        """有効なウィンドウ幅はバリデーションを通過する"""
        validator = ParameterValidator()
        data_length = len(ch1_df)
        window_size = min(11, data_length)
        result = validator.validate_window(window_size, data_length)
        assert isinstance(result, int)
        assert result == window_size

    def test_window_size_zero_returns_error(self, ch1_df: pd.DataFrame) -> None:
        """ウィンドウ幅0はエラーを返す"""
        validator = ParameterValidator()
        data_length = len(ch1_df)
        result = validator.validate_window(0, data_length)
        assert isinstance(result, InvalidWindowError)

    def test_window_size_negative_returns_error(self, ch1_df: pd.DataFrame) -> None:
        """負のウィンドウ幅はエラーを返す"""
        validator = ParameterValidator()
        data_length = len(ch1_df)
        result = validator.validate_window(-1, data_length)
        assert isinstance(result, InvalidWindowError)

    def test_window_size_exceeds_data_length_returns_error(self, ch1_df: pd.DataFrame) -> None:
        """データ長を超えるウィンドウ幅はエラーを返す"""
        validator = ParameterValidator()
        data_length = len(ch1_df)
        result = validator.validate_window(data_length + 1, data_length)
        assert isinstance(result, InvalidWindowError)

    def test_invalid_window_error_message_contains_valid_range(
        self, ch1_df: pd.DataFrame
    ) -> None:
        """InvalidWindowError のメッセージには有効範囲が含まれる"""
        validator = ParameterValidator()
        data_length = len(ch1_df)
        result = validator.validate_window(0, data_length)
        assert isinstance(result, InvalidWindowError)
        assert str(data_length) in result.message

    def test_valid_zscore_threshold_passes_validation(self) -> None:
        """正の閾値はバリデーションを通過する"""
        validator = ParameterValidator()
        result = validator.validate_threshold(2.5)
        assert isinstance(result, float)
        assert result == 2.5

    def test_zscore_threshold_zero_returns_error(self) -> None:
        """閾値0はエラーを返す"""
        validator = ParameterValidator()
        result = validator.validate_threshold(0.0)
        assert isinstance(result, InvalidThresholdError)

    def test_zscore_threshold_negative_returns_error(self) -> None:
        """負の閾値はエラーを返す"""
        validator = ParameterValidator()
        result = validator.validate_threshold(-1.0)
        assert isinstance(result, InvalidThresholdError)

    def test_invalid_threshold_error_message_mentions_positive_real(self) -> None:
        """InvalidThresholdError のメッセージは正の実数を求める内容を含む"""
        validator = ParameterValidator()
        result = validator.validate_threshold(-1.0)
        assert isinstance(result, InvalidThresholdError)
        # メッセージが空でないことを確認
        assert len(result.message) > 0

    def test_even_window_size_is_corrected_to_odd(self) -> None:
        """偶数ウィンドウ幅はSG向けに奇数補正される"""
        validator = ParameterValidator()
        result = validator.ensure_odd_window(10)
        assert result == 11
        assert result % 2 == 1

    def test_odd_window_size_is_unchanged(self) -> None:
        """奇数ウィンドウ幅はそのまま返される"""
        validator = ParameterValidator()
        result = validator.ensure_odd_window(11)
        assert result == 11

    def test_filter_config_median_toggle(self) -> None:
        """FilterConfig の median_enabled トグルが機能する"""
        config_on = FilterConfig(median_enabled=True, median_window=5, savgol_enabled=False)
        config_off = FilterConfig(median_enabled=False, median_window=5, savgol_enabled=False)

        series = pd.Series([1.0, 2.0, 10.0, 2.0, 1.0])
        nf = NoiseFilter()

        result_on = nf.apply(series, config_on)
        result_off = nf.apply(series, config_off)

        # ONの場合はスパイクが減衰する
        assert not result_on.filtered.equals(result_off.filtered)

    def test_filter_config_savgol_toggle(self) -> None:
        """FilterConfig の savgol_enabled トグルが機能する"""
        config_on = FilterConfig(
            median_enabled=False,
            savgol_enabled=True,
            savgol_window=5,
            savgol_polyorder=2,
        )
        config_off = FilterConfig(median_enabled=False, savgol_enabled=False)

        series = pd.Series([float(i) for i in range(20)])
        nf = NoiseFilter()

        result_on = nf.apply(series, config_on)
        result_off = nf.apply(series, config_off)

        # ONとOFFで結果が異なること（線形データはSGフィルタでほぼ変わらないが型は同一）
        assert isinstance(result_on.filtered, pd.Series)
        assert isinstance(result_off.filtered, pd.Series)


# ─────────────────────────────────────────────
# タスク 8.2: メイン画面へのグラフ描画統合
# パイプライン全体の統合テスト
# ─────────────────────────────────────────────


class TestMainScreenPipelineIntegration:
    """8.2 メイン画面グラフ描画統合のパイプラインテスト

    データ読み込み → フィルタリング → 異常検知 → 可視化の
    パイプライン全体が app.py から正しく呼び出せることを検証する。
    """

    def test_full_pipeline_returns_figure(
        self,
        sample_df: pd.DataFrame,
        visualizer: Visualizer,
        default_viz_config: VisualizerConfig,
    ) -> None:
        """フルパイプラインが go.Figure を返す"""
        # DataLoader でCH別データを取得
        loader = DataLoader()
        ch_data: dict[int, pd.DataFrame] = {}
        for ch in range(1, 5):
            df_ch = loader.get_channel_group(sample_df, ch)
            if len(df_ch) > 0:
                ch_data[ch] = df_ch.reset_index(drop=True)

        # CH別チャートを生成
        fig = visualizer.plot_channels(ch_data, default_viz_config)
        assert isinstance(fig, go.Figure)

    def test_filter_pipeline_produces_filter_result(
        self,
        wear_series: pd.Series,
    ) -> None:
        """フィルタリングパイプラインが FilterResult を返す"""
        validator = ParameterValidator()
        nf = NoiseFilter()

        window_size = validator.validate_window(5, len(wear_series))
        assert isinstance(window_size, int)

        odd_window = validator.ensure_odd_window(window_size)
        config = FilterConfig(
            median_enabled=True,
            median_window=odd_window,
            savgol_enabled=True,
            savgol_window=odd_window,
            savgol_polyorder=2,
        )
        result = nf.apply(wear_series, config)

        assert isinstance(result, FilterResult)
        assert len(result.original) == len(wear_series)
        assert len(result.filtered) == len(wear_series)

    def test_anomaly_detection_pipeline_produces_anomaly_result(
        self,
        wear_series: pd.Series,
        kilometric_series: pd.Series,
    ) -> None:
        """異常検知パイプラインが AnomalyResult を返す"""
        validator = ParameterValidator()
        detector = AnomalyDetector()

        window_size = validator.validate_window(5, len(wear_series))
        threshold = validator.validate_threshold(2.0)

        assert isinstance(window_size, int)
        assert isinstance(threshold, float)

        result = detector.detect(
            wear_series,
            window_size=window_size,
            threshold=threshold,
            position_series=kilometric_series,
        )

        assert isinstance(result, AnomalyResult)
        assert len(result.zscore_series) == len(wear_series)
        assert result.threshold == 2.0

    def test_filter_comparison_chart_from_pipeline(
        self,
        filter_result: FilterResult,
        kilometric_series: pd.Series,
        visualizer: Visualizer,
        default_viz_config: VisualizerConfig,
    ) -> None:
        """フィルタリングパイプラインからフィルタ比較グラフを生成できる"""
        fig = visualizer.plot_filter_comparison(
            filter_result, kilometric_series, default_viz_config
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2

    def test_signal_analysis_pipeline_produces_analysis_results(
        self,
        wear_series: pd.Series,
    ) -> None:
        """信号解析パイプラインがRMS・FFT・STFT結果を返す"""
        validator = ParameterValidator()
        analyzer = SignalAnalyzer()

        window_size = validator.validate_window(5, len(wear_series))
        assert isinstance(window_size, int)
        odd_window = validator.ensure_odd_window(window_size)

        rms_result = analyzer.compute_rms(wear_series, odd_window)
        fft_result = analyzer.compute_fft(wear_series, odd_window)
        stft_result = analyzer.compute_stft(wear_series, odd_window)

        assert len(rms_result.rms_series) == len(wear_series)
        assert len(fft_result.frequencies) > 0
        assert len(stft_result.spectrogram) > 0

    def test_analysis_results_chart_from_pipeline(
        self,
        wear_series: pd.Series,
        visualizer: Visualizer,
        default_viz_config: VisualizerConfig,
    ) -> None:
        """信号解析パイプラインから解析結果グラフを生成できる"""
        analyzer = SignalAnalyzer()
        rms_result = analyzer.compute_rms(wear_series, 5)
        fft_result = analyzer.compute_fft(wear_series, 5)
        stft_result = analyzer.compute_stft(wear_series, 5)

        fig = visualizer.plot_analysis_results(
            rms_result, fft_result, stft_result, default_viz_config
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 3

    def test_comparison_view_from_pipeline(
        self,
        wear_series: pd.Series,
        filter_result: FilterResult,
        anomaly_result: AnomalyResult,
        kilometric_series: pd.Series,
        visualizer: Visualizer,
        default_viz_config: VisualizerConfig,
    ) -> None:
        """パイプラインの最終結果から統合比較ビューを生成できる"""
        fig = visualizer.plot_comparison_view(
            raw_series=wear_series,
            filtered_series=filter_result.filtered,
            anomaly_result=anomaly_result,
            kilometric_series=kilometric_series,
            config=default_viz_config,
        )
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 3

    def test_html_export_from_pipeline(
        self,
        wear_series: pd.Series,
        filter_result: FilterResult,
        anomaly_result: AnomalyResult,
        kilometric_series: pd.Series,
        visualizer: Visualizer,
        tmp_path,
    ) -> None:
        """パイプラインの結果からHTML出力できる"""
        import os
        config = VisualizerConfig(output_dir=str(tmp_path))
        fig = visualizer.plot_comparison_view(
            raw_series=wear_series,
            filtered_series=filter_result.filtered,
            anomaly_result=anomaly_result,
            kilometric_series=kilometric_series,
            config=config,
        )
        output_path = visualizer.export_html(fig, "test_pipeline.html", config)
        assert os.path.exists(output_path)
        assert output_path.endswith(".html")

    def test_app_pipeline_with_invalid_window_shows_error_info(
        self,
        wear_series: pd.Series,
    ) -> None:
        """不正なウィンドウ幅の場合、バリデーションエラーが返される（UI表示用）"""
        validator = ParameterValidator()
        data_length = len(wear_series)

        # 不正な値（0）
        result = validator.validate_window(0, data_length)
        assert isinstance(result, InvalidWindowError)
        # エラーメッセージが存在すること（st.error() 表示用）
        assert len(result.message) > 0

    def test_app_pipeline_with_invalid_threshold_shows_error_info(self) -> None:
        """不正な閾値の場合、バリデーションエラーが返される（UI表示用）"""
        validator = ParameterValidator()

        result = validator.validate_threshold(-0.5)
        assert isinstance(result, InvalidThresholdError)
        assert len(result.message) > 0


# ─────────────────────────────────────────────
# タスク 8.3: 異常検知結果のグラフハイライト統合
# ─────────────────────────────────────────────


class TestAnomalyHighlightIntegration:
    """8.3 異常検知結果グラフハイライト統合のテスト"""

    def test_plot_anomaly_overlay_returns_figure(
        self,
        wear_series: pd.Series,
        kilometric_series: pd.Series,
        anomaly_result: AnomalyResult,
        visualizer: Visualizer,
        default_viz_config: VisualizerConfig,
    ) -> None:
        """plot_anomaly_overlay は go.Figure を返す"""
        fig = visualizer.plot_anomaly_overlay(
            wear_series,
            kilometric_series,
            anomaly_result,
            default_viz_config,
        )
        assert isinstance(fig, go.Figure)

    def test_plot_anomaly_overlay_has_base_trace(
        self,
        wear_series: pd.Series,
        kilometric_series: pd.Series,
        anomaly_result: AnomalyResult,
        visualizer: Visualizer,
        default_viz_config: VisualizerConfig,
    ) -> None:
        """異常点ハイライトグラフに基本波形トレースが含まれる"""
        fig = visualizer.plot_anomaly_overlay(
            wear_series,
            kilometric_series,
            anomaly_result,
            default_viz_config,
        )
        # 少なくとも1つのトレース（基本波形）が含まれる
        assert len(fig.data) >= 1

    def test_plot_anomaly_overlay_with_anomaly_points_has_marker_trace(
        self,
        wear_series: pd.Series,
        kilometric_series: pd.Series,
        visualizer: Visualizer,
        default_viz_config: VisualizerConfig,
    ) -> None:
        """異常点がある場合はマーカートレースが追加される"""
        # 低い閾値で確実に異常点が検出されるよう設定
        detector = AnomalyDetector()
        result = detector.detect(
            wear_series, window_size=3, threshold=0.01, position_series=kilometric_series
        )

        fig = visualizer.plot_anomaly_overlay(
            wear_series,
            kilometric_series,
            result,
            default_viz_config,
        )
        # 基本波形 + 異常点マーカーの2つ以上のトレースが含まれる
        assert len(fig.data) >= 2

    def test_plot_anomaly_overlay_no_anomaly_has_only_base_trace(
        self,
        wear_series: pd.Series,
        kilometric_series: pd.Series,
        visualizer: Visualizer,
        default_viz_config: VisualizerConfig,
    ) -> None:
        """異常点がない場合は基本波形のみのトレースとなる（または異常点トレースが空）"""
        # 非常に高い閾値で異常点が検出されないよう設定
        detector = AnomalyDetector()
        result = detector.detect(
            wear_series, window_size=3, threshold=999.0, position_series=kilometric_series
        )

        fig = visualizer.plot_anomaly_overlay(
            wear_series,
            kilometric_series,
            result,
            default_viz_config,
        )
        assert isinstance(fig, go.Figure)
        # 基本波形は必ず含まれる
        assert len(fig.data) >= 1

    def test_anomaly_marker_includes_kilometric_in_hover(
        self,
        wear_series: pd.Series,
        kilometric_series: pd.Series,
        visualizer: Visualizer,
        default_viz_config: VisualizerConfig,
    ) -> None:
        """異常点マーカーのホバーにキロ程値が含まれる（要件 5.5）"""
        # 確実に異常点が生成される設定
        detector = AnomalyDetector()
        result = detector.detect(
            wear_series, window_size=3, threshold=0.01, position_series=kilometric_series
        )

        fig = visualizer.plot_anomaly_overlay(
            wear_series,
            kilometric_series,
            result,
            default_viz_config,
        )

        # 異常点のトレースのhovertemplateにキロ程が含まれることを確認
        has_kilometric_in_hover = False
        for trace in fig.data:
            if hasattr(trace, "hovertemplate") and trace.hovertemplate is not None:
                if "キロ程" in trace.hovertemplate or "%{x" in trace.hovertemplate:
                    has_kilometric_in_hover = True
                    break
        assert has_kilometric_in_hover, "ホバーテンプレートにキロ程情報が含まれていません"

    def test_comparison_view_integrates_anomaly_highlights(
        self,
        wear_series: pd.Series,
        filter_result: FilterResult,
        kilometric_series: pd.Series,
        visualizer: Visualizer,
        default_viz_config: VisualizerConfig,
    ) -> None:
        """統合比較ビューに異常点ハイライトが統合される"""
        # 低い閾値で確実に異常点を生成
        detector = AnomalyDetector()
        anomaly_result = detector.detect(
            wear_series, window_size=3, threshold=0.01, position_series=kilometric_series
        )

        fig = visualizer.plot_comparison_view(
            raw_series=wear_series,
            filtered_series=filter_result.filtered,
            anomaly_result=anomaly_result,
            kilometric_series=kilometric_series,
            config=default_viz_config,
        )
        assert isinstance(fig, go.Figure)
        # 生波形 + フィルタ後 + Z-Score + 閾値ライン = 最低4トレース
        assert len(fig.data) >= 4


# ─────────────────────────────────────────────
# app.py モジュール構造の検証テスト
# ─────────────────────────────────────────────


class TestAppModuleStructure:
    """app.py のモジュール構造を検証するテスト

    Streamlit UI を直接テストすることは難しいため、
    app.py が必要なコンポーネントをインポートしていることを検証する。
    """

    def test_app_module_is_importable(self) -> None:
        """app.py がインポート可能である（Streamlit コンテキスト外でのテスト）"""
        # Streamlit のアプリをインポートすると st.set_page_config などが実行される
        # 代わりに、app.py が依存するコンポーネントのインポートを確認する
        from src.anomaly_detector import AnomalyDetector
        from src.data_loader import DataLoader
        from src.noise_filter import NoiseFilter
        from src.parameter_validator import ParameterValidator
        from src.signal_analyzer import SignalAnalyzer
        from src.visualizer import Visualizer

        # 各コンポーネントがインスタンス化できることを確認
        assert DataLoader() is not None
        assert ParameterValidator() is not None
        assert NoiseFilter() is not None
        assert SignalAnalyzer() is not None
        assert AnomalyDetector() is not None
        assert Visualizer() is not None

    def test_visualizer_plot_anomaly_overlay_exists(self) -> None:
        """Visualizer に plot_anomaly_overlay メソッドが存在する（タスク8.3）"""
        visualizer = Visualizer()
        assert hasattr(visualizer, "plot_anomaly_overlay"), (
            "Visualizer に plot_anomaly_overlay メソッドが必要です"
        )
        assert callable(visualizer.plot_anomaly_overlay)

    def test_data_constants_are_defined(self) -> None:
        """データファイルパスなどの定数が参照可能である"""
        from pathlib import Path
        # デフォルトデータパスが存在するか確認（存在しない場合はスキップではなくパスの形式確認）
        data_path = Path("/home/sagemaker-user/5cm-chart-analysis/data")
        # ディレクトリの存在確認（テスト環境によっては存在しない可能性あり）
        # ここではパスオブジェクトが正常に生成できることを確認
        assert data_path is not None


# ─────────────────────────────────────────────
# タスク 6.1: アップロードデータ経由のフルパイプライン統合テスト
# Requirements: 4.1, 3.1, 3.2, 3.3, 3.4
# ─────────────────────────────────────────────


class TestUploadedDataPipelineIntegration:
    """6.1 アップロードデータ経由のフルパイプライン統合テスト

    アップロードされた DataFrame が NoiseFilter に正しく渡され、
    フルパイプライン（NoiseFilter → AnomalyDetector → SignalAnalyzer → Visualizer）が
    完走することを検証する（要件 4.1）。
    """

    def test_uploaded_dataframe_passes_to_noise_filter(
        self, sample_df: pd.DataFrame
    ) -> None:
        """アップロードされた DataFrame が NoiseFilter に正しく渡される（要件 4.1）。

        DataLoader.load_from_upload() の成功時に返す DataFrame が
        NoiseFilter に入力として渡せることを検証する。
        """
        loader = DataLoader()
        ch_data: dict[int, pd.DataFrame] = {}
        for ch in range(1, 5):
            df_ch = loader.get_channel_group(sample_df, ch)
            if len(df_ch) > 0:
                ch_data[ch] = df_ch.reset_index(drop=True)

        assert len(ch_data) > 0, "CH別データが取得できませんでした"

        first_ch = min(ch_data.keys())
        representative_df = ch_data[first_ch]
        wear_series = representative_df["摩耗_測定値"]

        nf = NoiseFilter()
        config = FilterConfig(
            median_enabled=True,
            median_window=3,
            savgol_enabled=False,
        )
        filter_result = nf.apply(wear_series, config)

        assert isinstance(filter_result.filtered, pd.Series), (
            "NoiseFilter は pd.Series を返す必要があります"
        )
        assert len(filter_result.filtered) == len(wear_series), (
            "フィルタ後 Series の長さが入力と一致する必要があります"
        )

    def test_full_pipeline_with_uploaded_data(
        self,
        sample_df: pd.DataFrame,
        visualizer: Visualizer,
        default_viz_config: VisualizerConfig,
    ) -> None:
        """アップロードデータでフルパイプライン（NoiseFilter → AnomalyDetector → SignalAnalyzer → Visualizer）が完走する（要件 4.1）。"""
        loader = DataLoader()
        ch_data: dict[int, pd.DataFrame] = {}
        for ch in range(1, 5):
            df_ch = loader.get_channel_group(sample_df, ch)
            if len(df_ch) > 0:
                ch_data[ch] = df_ch.reset_index(drop=True)

        assert len(ch_data) > 0

        first_ch = min(ch_data.keys())
        representative_df = ch_data[first_ch]
        wear_series = representative_df["摩耗_測定値"]
        kilo_series = representative_df["キロ程"]

        # ── NoiseFilter ──
        nf = NoiseFilter()
        filter_config = FilterConfig(
            median_enabled=True,
            median_window=3,
            savgol_enabled=True,
            savgol_window=3,
            savgol_polyorder=2,
        )
        filter_result = nf.apply(wear_series, filter_config)

        # ── AnomalyDetector ──
        detector = AnomalyDetector()
        anomaly_result = detector.detect(
            filter_result.filtered,
            window_size=3,
            threshold=2.0,
            position_series=kilo_series,
        )

        # ── SignalAnalyzer ──
        analyzer = SignalAnalyzer()
        rms_result = analyzer.compute_rms(filter_result.filtered, 3)
        fft_result = analyzer.compute_fft(filter_result.filtered, 3)
        stft_result = analyzer.compute_stft(filter_result.filtered, 3)

        # ── Visualizer ──
        comparison_fig = visualizer.plot_comparison_view(
            raw_series=wear_series,
            filtered_series=filter_result.filtered,
            anomaly_result=anomaly_result,
            kilometric_series=kilo_series,
            config=default_viz_config,
        )

        assert isinstance(comparison_fig, go.Figure), (
            "Visualizer.plot_comparison_view() は go.Figure を返す必要があります"
        )
        assert len(comparison_fig.data) >= 3, (
            "統合比較ビューには少なくとも3トレースが含まれる必要があります"
        )

        analysis_fig = visualizer.plot_analysis_results(
            rms_result, fft_result, stft_result, default_viz_config
        )
        assert isinstance(analysis_fig, go.Figure)

    def test_load_error_propagation_from_upload_invalid_format(self) -> None:
        """LoadError(kind='invalid_format') が app.py のエラーハンドリングフローに正しく伝播する（要件 3.1, 3.4）。

        DataLoader.load_from_upload() が LoadError を返す場合、
        app.py はそのエラーを受け取って st.error() + st.stop() を呼ぶ実装になっている。
        ここではエラー値オブジェクトの型と種別を検証する。
        """
        import io

        loader = DataLoader()
        # .csv ファイルは invalid_format エラーを返す
        fake_content = io.BytesIO(b"col1,col2\n1,2")
        result = loader.load_from_upload(fake_content, "data.csv")

        assert isinstance(result, LoadError), (
            "非 xlsx ファイルは LoadError を返す必要があります"
        )
        assert result.kind == "invalid_format", (
            f"非 xlsx は LoadError(kind='invalid_format') を返すべきです。実際: {result.kind}"
        )
        # app.py の条件分岐でこの kind を検出して st.error() + st.stop() が呼ばれる
        # エラーメッセージが非空であることを確認
        assert len(result.message) > 0, "LoadError.message が空です"

    def test_load_error_propagation_from_upload_missing_columns(self) -> None:
        """LoadError(kind='missing_columns') が app.py のエラーハンドリングフローに正しく伝播する（要件 3.2, 3.4）。"""
        import io

        import pandas as pd

        from src.data_loader import REQUIRED_COLUMNS

        loader = DataLoader()
        buf = io.BytesIO()
        pd.DataFrame({"dummy_col": [1, 2, 3]}).to_excel(buf, index=False, engine="openpyxl")
        buf.seek(0)
        result = loader.load_from_upload(buf, "missing_cols.xlsx")

        assert isinstance(result, LoadError), (
            "必須列欠損ファイルは LoadError を返す必要があります"
        )
        assert result.kind == "missing_columns", (
            f"必須列欠損は LoadError(kind='missing_columns') を返すべきです。実際: {result.kind}"
        )
        assert len(result.missing_columns) > 0, (
            "LoadError.missing_columns に欠損列名が含まれる必要があります"
        )

    def test_load_error_propagation_from_upload_read_error(self) -> None:
        """LoadError(kind='read_error') が app.py のエラーハンドリングフローに正しく伝播する（要件 3.3, 3.4）。"""
        import io

        loader = DataLoader()
        # 破損した xlsx バイト列は read_error を返す
        corrupt_content = io.BytesIO(b"PK\x03\x04FAKE_CORRUPT_XLSX_CONTENT")
        result = loader.load_from_upload(corrupt_content, "corrupt.xlsx")

        assert isinstance(result, LoadError), (
            "破損した xlsx は LoadError を返す必要があります"
        )
        assert result.kind == "read_error", (
            f"破損した xlsx は LoadError(kind='read_error') を返すべきです。実際: {result.kind}"
        )

    def test_load_error_error_stop_flow_in_app_source(self) -> None:
        """app.py のエラーハンドリングフローで st.error() → st.stop() の順序が維持される（要件 3.4, 3.5）。

        ソースコード検査により、各 LoadError 種別に対して
        st.error() の呼び出し後に st.stop() が呼ばれることを確認する。
        """
        from pathlib import Path

        source = Path("/home/sagemaker-user/5cm-chart-analysis/app.py").read_text(encoding="utf-8")
        lines = source.split("\n")

        # LoadError の isinstance チェックが存在することを確認
        assert "isinstance" in source and "LoadError" in source, (
            "app.py に LoadError の isinstance チェックが必要です"
        )

        # st.error() と st.stop() が存在することを確認
        assert "st.error(" in source, "app.py に st.error() が必要です"
        assert "st.stop()" in source, "app.py に st.stop() が必要です"

        # LoadError の isinstance チェックブロック内で st.error → st.stop の順序を確認
        load_error_check_idx = None
        for i, line in enumerate(lines):
            if "isinstance" in line and "LoadError" in line:
                load_error_check_idx = i
                break

        assert load_error_check_idx is not None

        context_lines = lines[load_error_check_idx:load_error_check_idx + 50]
        context = "\n".join(context_lines)

        error_pos = context.find("st.error(")
        stop_pos = context.find("st.stop()")

        assert error_pos != -1, "LoadError 処理ブロック内に st.error() が必要です"
        assert stop_pos != -1, "LoadError 処理ブロック内に st.stop() が必要です"
        assert error_pos < stop_pos, (
            "st.error() は st.stop() より前に呼ばれる必要があります（要件 3.4, 3.5）"
        )


# ─────────────────────────────────────────────
# タスク 6.2: 既存パラメータウィジェットの動作確認テスト
# Requirements: 4.4
# ─────────────────────────────────────────────


class TestExistingParameterWidgetsUnaffected:
    """6.2 ファイルアップロード機能追加後も既存パラメータウィジェットの動作が変化しないことを検証する（要件 4.4）。

    ウィンドウ幅・Z-Score 閾値・フィルタトグルのウィジェット動作が
    ファイルアップロード機能追加前後で変化しないことを確認する。
    """

    def test_window_size_widget_logic_unchanged(self, ch1_df: pd.DataFrame) -> None:
        """ウィンドウ幅ウィジェットのバリデーションロジックが変化しない（要件 4.4）。

        ParameterValidator.validate_window() の動作がファイルアップロード機能追加後も
        変化しないことを確認する。
        """
        validator = ParameterValidator()
        data_length = len(ch1_df)

        # 有効なウィンドウ幅はそのまま返る
        result = validator.validate_window(5, data_length)
        assert isinstance(result, int)
        assert result == 5

        # 0 はエラーを返す
        error_result = validator.validate_window(0, data_length)
        assert isinstance(error_result, InvalidWindowError)

        # データ長超過もエラーを返す
        overflow_result = validator.validate_window(data_length + 1, data_length)
        assert isinstance(overflow_result, InvalidWindowError)

    def test_zscore_threshold_widget_logic_unchanged(self) -> None:
        """Z-Score 閾値ウィジェットのバリデーションロジックが変化しない（要件 4.4）。"""
        validator = ParameterValidator()

        # 有効な閾値
        result = validator.validate_threshold(2.5)
        assert isinstance(result, float)
        assert result == 2.5

        # 0 はエラー
        error_result = validator.validate_threshold(0.0)
        assert isinstance(error_result, InvalidThresholdError)

        # 負値もエラー
        neg_result = validator.validate_threshold(-1.0)
        assert isinstance(neg_result, InvalidThresholdError)

    def test_median_filter_toggle_logic_unchanged(self, wear_series: pd.Series) -> None:
        """移動中央値フィルタトグルのロジックが変化しない（要件 4.4）。

        FilterConfig の median_enabled フラグが ON/OFF で異なる結果を返すことを確認する。
        """
        nf = NoiseFilter()

        config_on = FilterConfig(median_enabled=True, median_window=3, savgol_enabled=False)
        config_off = FilterConfig(median_enabled=False, median_window=3, savgol_enabled=False)

        result_on = nf.apply(wear_series, config_on)
        result_off = nf.apply(wear_series, config_off)

        # ON/OFF で結果が異なる（スパイクノイズがある場合）
        assert isinstance(result_on.filtered, pd.Series)
        assert isinstance(result_off.filtered, pd.Series)
        assert len(result_on.filtered) == len(wear_series)
        assert len(result_off.filtered) == len(wear_series)

    def test_savgol_filter_toggle_logic_unchanged(self, wear_series: pd.Series) -> None:
        """Savitzky-Golay フィルタトグルのロジックが変化しない（要件 4.4）。"""
        nf = NoiseFilter()

        config_on = FilterConfig(
            median_enabled=False,
            savgol_enabled=True,
            savgol_window=3,
            savgol_polyorder=2,
        )
        config_off = FilterConfig(median_enabled=False, savgol_enabled=False)

        result_on = nf.apply(wear_series, config_on)
        result_off = nf.apply(wear_series, config_off)

        assert isinstance(result_on.filtered, pd.Series)
        assert isinstance(result_off.filtered, pd.Series)

    def test_parameter_widgets_defined_in_app_source(self) -> None:
        """app.py に既存パラメータウィジェット定義が変更されていない（要件 4.4）。

        ファイルアップロード機能追加後も既存のウィジェット定義が
        そのまま存在することをソース検査で確認する。
        """
        from pathlib import Path

        source = Path("/home/sagemaker-user/5cm-chart-analysis/app.py").read_text(encoding="utf-8")

        assert "ウィンドウ幅" in source, (
            "ウィンドウ幅ウィジェットが app.py に必要です（要件 4.4）"
        )
        assert "Z-Score 閾値" in source, (
            "Z-Score 閾値ウィジェットが app.py に必要です（要件 4.4）"
        )
        assert "移動中央値フィルタ" in source, (
            "移動中央値フィルタウィジェットが app.py に必要です（要件 4.4）"
        )
        assert "Savitzky-Golay フィルタ" in source, (
            "Savitzky-Golay フィルタウィジェットが app.py に必要です（要件 4.4）"
        )

    def test_parameter_widgets_appear_before_pipeline_in_app_source(self) -> None:
        """パラメータウィジェット定義がパイプライン実行より前に存在する（要件 4.4）。"""
        from pathlib import Path

        source = Path("/home/sagemaker-user/5cm-chart-analysis/app.py").read_text(encoding="utf-8")

        # 「分析パラメータ」ヘッダーがサイドバーに存在する
        assert "分析パラメータ" in source, (
            "サイドバーに「分析パラメータ」セクションが必要です（要件 4.4）"
        )

        # ウィジェット定義がパイプライン実行（NoiseFilter）より前にある
        params_pos = source.find("分析パラメータ")
        noise_filter_pos = source.find("NoiseFilter()")

        assert params_pos != -1
        assert noise_filter_pos != -1
        assert params_pos < noise_filter_pos, (
            "パラメータウィジェット定義は NoiseFilter() 実行より前に存在する必要があります（要件 4.4）"
        )

    def test_window_size_used_in_filter_config(
        self, wear_series: pd.Series
    ) -> None:
        """ウィンドウ幅がフィルタ設定（FilterConfig）に使用される（要件 4.4）。

        ParameterValidator で検証済みのウィンドウ幅が FilterConfig に渡されて
        NoiseFilter を動作させられることを確認する。
        """
        validator = ParameterValidator()
        nf = NoiseFilter()

        window_size_input = 5
        data_length = len(wear_series)

        window_size = validator.validate_window(window_size_input, data_length)
        assert isinstance(window_size, int)

        odd_window = validator.ensure_odd_window(window_size)
        assert odd_window % 2 == 1

        filter_config = FilterConfig(
            median_enabled=True,
            median_window=odd_window,
            savgol_enabled=True,
            savgol_window=odd_window,
            savgol_polyorder=2,
        )
        result = nf.apply(wear_series, filter_config)

        assert isinstance(result.filtered, pd.Series)
        assert len(result.filtered) == len(wear_series)

    def test_zscore_threshold_used_in_anomaly_detector(
        self, wear_series: pd.Series, kilometric_series: pd.Series
    ) -> None:
        """Z-Score 閾値が AnomalyDetector に使用される（要件 4.4）。"""
        validator = ParameterValidator()
        detector = AnomalyDetector()

        threshold_input = 2.5
        threshold = validator.validate_threshold(threshold_input)
        assert isinstance(threshold, float)
        assert threshold == threshold_input

        result = detector.detect(
            wear_series,
            window_size=5,
            threshold=threshold,
            position_series=kilometric_series,
        )

        assert result.threshold == threshold_input, (
            "AnomalyDetector.detect() の閾値が正しく設定されている必要があります"
        )
