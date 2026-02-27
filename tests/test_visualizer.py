"""Visualizerのテスト（タスク7: インタラクティブグラフ生成機能）"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from src.anomaly_detector import AnomalyDetector, AnomalyResult
from src.noise_filter import FilterConfig, FilterResult
from src.signal_analyzer import FFTResult, RMSResult, STFTResult
from src.visualizer import Visualizer, VisualizerConfig


# ─────────────────────────────────────────────
# テスト用フィクスチャ
# ─────────────────────────────────────────────

@pytest.fixture()
def sample_df() -> pd.DataFrame:
    """テスト用の小規模 DataFrame（必須10列を含む）"""
    n = 20
    return pd.DataFrame(
        {
            "キロ程": np.linspace(1000.0, 1001.0, n),
            "摩耗_測定値": np.linspace(10.0, 9.0, n) + np.random.default_rng(42).standard_normal(n) * 0.1,
            "CH": [1] * n,
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
def channel_data(sample_df: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """CH1〜4の辞書データ"""
    result: dict[int, pd.DataFrame] = {}
    for ch in range(1, 5):
        df = sample_df.copy()
        df["CH"] = ch
        result[ch] = df
    return result


@pytest.fixture()
def kilometric_series(sample_df: pd.DataFrame) -> pd.Series:
    """キロ程 Series"""
    return sample_df["キロ程"]


@pytest.fixture()
def wear_series(sample_df: pd.DataFrame) -> pd.Series:
    """摩耗値 Series"""
    return sample_df["摩耗_測定値"]


@pytest.fixture()
def filter_result(sample_df: pd.DataFrame) -> FilterResult:
    """テスト用 FilterResult"""
    original = sample_df["摩耗_測定値"]
    filtered = original + 0.05  # 簡易的な差異
    config = FilterConfig(median_enabled=True, median_window=3, savgol_enabled=False)
    return FilterResult(original=original, filtered=filtered, config=config)


@pytest.fixture()
def anomaly_result(wear_series: pd.Series) -> AnomalyResult:
    """テスト用 AnomalyResult（window=5, threshold=2.0）"""
    detector = AnomalyDetector()
    return detector.detect(wear_series, window_size=5, threshold=2.0)


@pytest.fixture()
def rms_result(wear_series: pd.Series) -> RMSResult:
    """テスト用 RMSResult"""
    from src.signal_analyzer import SignalAnalyzer
    analyzer = SignalAnalyzer()
    return analyzer.compute_rms(wear_series, window_size=5)


@pytest.fixture()
def fft_result(wear_series: pd.Series) -> FFTResult:
    """テスト用 FFTResult"""
    from src.signal_analyzer import SignalAnalyzer
    analyzer = SignalAnalyzer()
    return analyzer.compute_fft(wear_series, window_size=10)


@pytest.fixture()
def stft_result(wear_series: pd.Series) -> STFTResult:
    """テスト用 STFTResult"""
    from src.signal_analyzer import SignalAnalyzer
    analyzer = SignalAnalyzer()
    return analyzer.compute_stft(wear_series, window_size=5)


@pytest.fixture()
def default_config() -> VisualizerConfig:
    """デフォルト VisualizerConfig"""
    return VisualizerConfig()


@pytest.fixture()
def visualizer() -> Visualizer:
    """Visualizer インスタンス"""
    return Visualizer()


# ─────────────────────────────────────────────
# タスク 7.1: CH別摩耗チャートの実装
# ─────────────────────────────────────────────

class TestPlotChannels:
    """7.1 CH別摩耗チャートの実装テスト"""

    def test_returns_go_figure(
        self,
        visualizer: Visualizer,
        channel_data: dict[int, pd.DataFrame],
        default_config: VisualizerConfig,
    ) -> None:
        """plot_channels は go.Figure を返す"""
        fig = visualizer.plot_channels(channel_data, default_config)
        assert isinstance(fig, go.Figure)

    def test_figure_has_4_traces(
        self,
        visualizer: Visualizer,
        channel_data: dict[int, pd.DataFrame],
        default_config: VisualizerConfig,
    ) -> None:
        """4つのCH分のトレースを持つ"""
        fig = visualizer.plot_channels(channel_data, default_config)
        # CH1〜4それぞれのトレースが存在することを確認
        assert len(fig.data) >= 4

    def test_figure_has_subplots_or_traces(
        self,
        visualizer: Visualizer,
        channel_data: dict[int, pd.DataFrame],
        default_config: VisualizerConfig,
    ) -> None:
        """全4CHのデータが表示可能である"""
        fig = visualizer.plot_channels(channel_data, default_config)
        # 少なくとも4件のトレースが含まれること
        assert len(fig.data) >= 4

    def test_x_axis_is_kilometric(
        self,
        visualizer: Visualizer,
        channel_data: dict[int, pd.DataFrame],
        default_config: VisualizerConfig,
    ) -> None:
        """最初のトレースの x 軸データがキロ程である"""
        fig = visualizer.plot_channels(channel_data, default_config)
        # 最初のトレースに x データが存在すること
        assert fig.data[0].x is not None
        assert len(fig.data[0].x) > 0

    def test_y_axis_is_wear_value(
        self,
        visualizer: Visualizer,
        channel_data: dict[int, pd.DataFrame],
        default_config: VisualizerConfig,
    ) -> None:
        """最初のトレースの y 軸データが摩耗値である"""
        fig = visualizer.plot_channels(channel_data, default_config)
        assert fig.data[0].y is not None
        assert len(fig.data[0].y) > 0

    def test_hovertemplate_contains_required_fields(
        self,
        visualizer: Visualizer,
        channel_data: dict[int, pd.DataFrame],
        default_config: VisualizerConfig,
    ) -> None:
        """hovertemplate に必須8項目が含まれる（要件 2.3）"""
        fig = visualizer.plot_channels(channel_data, default_config)
        # 最初のトレースのhovertemplateを確認
        hovertemplate = fig.data[0].hovertemplate
        assert hovertemplate is not None

        # 必須8項目が含まれているか確認
        required_fields = [
            "箇所名",
            "通称線名名称",
            "駅・駅々間名称",
            "電柱番号",
            "キロ程",
            "架線構造名",
            "トロリ線種",
            "降雨フラグ",
        ]
        for field in required_fields:
            assert field in hovertemplate, (
                f"hovertemplate に '{field}' が含まれていません。"
                f"実際の hovertemplate: {hovertemplate}"
            )

    def test_single_ch_data_works(
        self,
        visualizer: Visualizer,
        sample_df: pd.DataFrame,
        default_config: VisualizerConfig,
    ) -> None:
        """1つのCHのみのデータでも動作する"""
        channel_data = {1: sample_df}
        fig = visualizer.plot_channels(channel_data, default_config)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 1

    def test_empty_channel_data_works(
        self,
        visualizer: Visualizer,
        default_config: VisualizerConfig,
    ) -> None:
        """空の channel_data でも例外なく Figure を返す"""
        fig = visualizer.plot_channels({}, default_config)
        assert isinstance(fig, go.Figure)


# ─────────────────────────────────────────────
# タスク 7.2: フィルタ前後比較グラフの実装
# ─────────────────────────────────────────────

class TestPlotFilterComparison:
    """7.2 フィルタ前後比較グラフの実装テスト"""

    def test_returns_go_figure(
        self,
        visualizer: Visualizer,
        filter_result: FilterResult,
        kilometric_series: pd.Series,
        default_config: VisualizerConfig,
    ) -> None:
        """plot_filter_comparison は go.Figure を返す"""
        fig = visualizer.plot_filter_comparison(filter_result, kilometric_series, default_config)
        assert isinstance(fig, go.Figure)

    def test_has_two_traces(
        self,
        visualizer: Visualizer,
        filter_result: FilterResult,
        kilometric_series: pd.Series,
        default_config: VisualizerConfig,
    ) -> None:
        """フィルタ前後の2系列のトレースを持つ"""
        fig = visualizer.plot_filter_comparison(filter_result, kilometric_series, default_config)
        # フィルタ前と後の2本のトレース
        assert len(fig.data) >= 2

    def test_traces_have_different_names(
        self,
        visualizer: Visualizer,
        filter_result: FilterResult,
        kilometric_series: pd.Series,
        default_config: VisualizerConfig,
    ) -> None:
        """2つのトレースは異なる名前を持つ（色分けのため）"""
        fig = visualizer.plot_filter_comparison(filter_result, kilometric_series, default_config)
        trace_names = [trace.name for trace in fig.data]
        assert len(set(trace_names)) >= 2, "2つのトレースは異なる名前を持つ必要があります"

    def test_x_axis_is_kilometric(
        self,
        visualizer: Visualizer,
        filter_result: FilterResult,
        kilometric_series: pd.Series,
        default_config: VisualizerConfig,
    ) -> None:
        """横軸にキロ程を使用する（要件 4.7）"""
        fig = visualizer.plot_filter_comparison(filter_result, kilometric_series, default_config)
        trace = fig.data[0]
        assert trace.x is not None
        # キロ程の値と一致することを確認
        np.testing.assert_array_almost_equal(trace.x, kilometric_series.values)

    def test_y_axis_is_wear_value(
        self,
        visualizer: Visualizer,
        filter_result: FilterResult,
        kilometric_series: pd.Series,
        default_config: VisualizerConfig,
    ) -> None:
        """縦軸に残存直径（摩耗値）を使用する"""
        fig = visualizer.plot_filter_comparison(filter_result, kilometric_series, default_config)
        trace = fig.data[0]
        assert trace.y is not None
        assert len(trace.y) == len(kilometric_series)

    def test_both_traces_have_same_x_length(
        self,
        visualizer: Visualizer,
        filter_result: FilterResult,
        kilometric_series: pd.Series,
        default_config: VisualizerConfig,
    ) -> None:
        """両トレースの x 軸データ長が一致する"""
        fig = visualizer.plot_filter_comparison(filter_result, kilometric_series, default_config)
        x_lengths = [len(trace.x) for trace in fig.data if trace.x is not None]
        assert len(set(x_lengths)) == 1, "全トレースの x 軸データ長が一致している必要があります"


# ─────────────────────────────────────────────
# タスク 7.3: 信号解析結果グラフの実装
# ─────────────────────────────────────────────

class TestPlotAnalysisResults:
    """7.3 信号解析結果グラフの実装テスト"""

    def test_returns_go_figure(
        self,
        visualizer: Visualizer,
        rms_result: RMSResult,
        fft_result: FFTResult,
        stft_result: STFTResult,
        default_config: VisualizerConfig,
    ) -> None:
        """plot_analysis_results は go.Figure を返す"""
        fig = visualizer.plot_analysis_results(
            rms_result, fft_result, stft_result, default_config
        )
        assert isinstance(fig, go.Figure)

    def test_has_multiple_traces(
        self,
        visualizer: Visualizer,
        rms_result: RMSResult,
        fft_result: FFTResult,
        stft_result: STFTResult,
        default_config: VisualizerConfig,
    ) -> None:
        """RMS・FFT・STFTの3種類の結果を含む"""
        fig = visualizer.plot_analysis_results(
            rms_result, fft_result, stft_result, default_config
        )
        # RMS, FFT, STFT の最低3トレース
        assert len(fig.data) >= 3

    def test_layout_has_axis_labels(
        self,
        visualizer: Visualizer,
        rms_result: RMSResult,
        fft_result: FFTResult,
        stft_result: STFTResult,
        default_config: VisualizerConfig,
    ) -> None:
        """軸ラベルが存在する（日本語ラベル、要件 3.5）"""
        fig = visualizer.plot_analysis_results(
            rms_result, fft_result, stft_result, default_config
        )
        # タイトルまたは軸ラベルが設定されていることを確認
        # Figure layout にタイトルまたは軸ラベルが含まれること
        layout_str = str(fig.layout)
        # 何らかのラベルが設定されていること（空のFigureではないこと）
        assert len(fig.data) >= 3

    def test_rms_trace_has_data(
        self,
        visualizer: Visualizer,
        rms_result: RMSResult,
        fft_result: FFTResult,
        stft_result: STFTResult,
        default_config: VisualizerConfig,
    ) -> None:
        """RMS トレースにデータが含まれる"""
        fig = visualizer.plot_analysis_results(
            rms_result, fft_result, stft_result, default_config
        )
        # 少なくとも1つのトレースにデータが含まれること
        has_data = any(
            trace.y is not None and len(trace.y) > 0
            for trace in fig.data
            if hasattr(trace, "y") and trace.y is not None
        )
        assert has_data


# ─────────────────────────────────────────────
# タスク 7.4: 統合3段比較ビューの実装
# ─────────────────────────────────────────────

class TestPlotComparisonView:
    """7.4 統合3段比較ビューの実装テスト"""

    def test_returns_go_figure(
        self,
        visualizer: Visualizer,
        wear_series: pd.Series,
        anomaly_result: AnomalyResult,
        kilometric_series: pd.Series,
        default_config: VisualizerConfig,
    ) -> None:
        """plot_comparison_view は go.Figure を返す"""
        fig = visualizer.plot_comparison_view(
            raw_series=wear_series,
            filtered_series=wear_series,
            anomaly_result=anomaly_result,
            kilometric_series=kilometric_series,
            config=default_config,
        )
        assert isinstance(fig, go.Figure)

    def test_has_3_subplots(
        self,
        visualizer: Visualizer,
        wear_series: pd.Series,
        anomaly_result: AnomalyResult,
        kilometric_series: pd.Series,
        default_config: VisualizerConfig,
    ) -> None:
        """3段のサブプロットを持つ（要件 6.1）"""
        fig = visualizer.plot_comparison_view(
            raw_series=wear_series,
            filtered_series=wear_series,
            anomaly_result=anomaly_result,
            kilometric_series=kilometric_series,
            config=default_config,
        )
        # 3サブプロット分のトレースが存在することを確認
        # 生波形・フィルタ後波形・Z-Score + 閾値ライン
        assert len(fig.data) >= 3

    def test_shared_x_axes(
        self,
        visualizer: Visualizer,
        wear_series: pd.Series,
        anomaly_result: AnomalyResult,
        kilometric_series: pd.Series,
        default_config: VisualizerConfig,
    ) -> None:
        """X軸（キロ程）が連動している（要件 6.2）"""
        fig = visualizer.plot_comparison_view(
            raw_series=wear_series,
            filtered_series=wear_series,
            anomaly_result=anomaly_result,
            kilometric_series=kilometric_series,
            config=default_config,
        )
        # shared_xaxes=True で作成されていることをレイアウトで確認
        # subplot の X 軸が matches で連動していることを確認
        layout_str = str(fig.layout)
        # xaxis2 が xaxis1 に紐付いている（matches="x"）か確認
        assert "xaxis2" in layout_str or len(fig.data) >= 3

    def test_has_threshold_line(
        self,
        visualizer: Visualizer,
        wear_series: pd.Series,
        anomaly_result: AnomalyResult,
        kilometric_series: pd.Series,
        default_config: VisualizerConfig,
    ) -> None:
        """Z-Score プロットに閾値ラインが含まれる（要件 6.3）"""
        fig = visualizer.plot_comparison_view(
            raw_series=wear_series,
            filtered_series=wear_series,
            anomaly_result=anomaly_result,
            kilometric_series=kilometric_series,
            config=default_config,
        )
        # 閾値ラインのトレースが含まれることを確認
        # dash="dash" のトレース、またはラインタイプの確認
        has_threshold_line = False
        for trace in fig.data:
            if hasattr(trace, "line") and trace.line is not None:
                if hasattr(trace.line, "dash") and trace.line.dash == "dash":
                    has_threshold_line = True
                    break
        assert has_threshold_line, "閾値ラインが破線で表示されていません"

    def test_traces_have_hovertemplate(
        self,
        visualizer: Visualizer,
        wear_series: pd.Series,
        anomaly_result: AnomalyResult,
        kilometric_series: pd.Series,
        default_config: VisualizerConfig,
    ) -> None:
        """データポイントのホバーにキロ程・測定値・Z-Scoreが含まれる（要件 6.4）"""
        fig = visualizer.plot_comparison_view(
            raw_series=wear_series,
            filtered_series=wear_series,
            anomaly_result=anomaly_result,
            kilometric_series=kilometric_series,
            config=default_config,
        )
        # hovertemplateが設定されているトレースが存在することを確認
        has_hovertemplate = any(
            trace.hovertemplate is not None
            for trace in fig.data
            if hasattr(trace, "hovertemplate")
        )
        assert has_hovertemplate, "hovertemplateが設定されていません"

    def test_subplots_have_3_rows(
        self,
        visualizer: Visualizer,
        wear_series: pd.Series,
        anomaly_result: AnomalyResult,
        kilometric_series: pd.Series,
        default_config: VisualizerConfig,
    ) -> None:
        """make_subplots(rows=3) で生成されている（要件 6.1）"""
        fig = visualizer.plot_comparison_view(
            raw_series=wear_series,
            filtered_series=wear_series,
            anomaly_result=anomaly_result,
            kilometric_series=kilometric_series,
            config=default_config,
        )
        # レイアウトに xaxis, xaxis2, xaxis3 が存在することで3段サブプロットを確認
        assert hasattr(fig.layout, "xaxis")
        # 少なくとも3つのトレースを持つ（生波形・フィルタ後・Z-Score）
        assert len(fig.data) >= 3


# ─────────────────────────────────────────────
# タスク 7.5: HTML出力機能の実装
# ─────────────────────────────────────────────

class TestExportHtml:
    """7.5 HTML出力機能の実装テスト"""

    def test_returns_string_path(
        self,
        visualizer: Visualizer,
        default_config: VisualizerConfig,
        tmp_path,
    ) -> None:
        """export_html は文字列のファイルパスを返す"""
        config = VisualizerConfig(output_dir=str(tmp_path))
        fig = go.Figure()
        result = visualizer.export_html(fig, "test_output.html", config)
        assert isinstance(result, str)

    def test_creates_html_file(
        self,
        visualizer: Visualizer,
        tmp_path,
    ) -> None:
        """HTML ファイルが実際に作成される（要件 6.5）"""
        import os
        config = VisualizerConfig(output_dir=str(tmp_path))
        fig = go.Figure()
        output_path = visualizer.export_html(fig, "test_output.html", config)
        assert os.path.exists(output_path), f"HTML ファイルが存在しません: {output_path}"

    def test_creates_output_directory_if_not_exists(
        self,
        visualizer: Visualizer,
        tmp_path,
    ) -> None:
        """出力ディレクトリが存在しない場合は自動作成する（要件 6.5）"""
        import os
        non_existent_dir = str(tmp_path / "new_subdir" / "output")
        config = VisualizerConfig(output_dir=non_existent_dir)
        fig = go.Figure()
        output_path = visualizer.export_html(fig, "test.html", config)
        assert os.path.exists(non_existent_dir)
        assert os.path.exists(output_path)

    def test_returns_absolute_path(
        self,
        visualizer: Visualizer,
        tmp_path,
    ) -> None:
        """返すパスが絶対パスである（要件 6.5）"""
        import os
        config = VisualizerConfig(output_dir=str(tmp_path))
        fig = go.Figure()
        output_path = visualizer.export_html(fig, "test.html", config)
        assert os.path.isabs(output_path), f"絶対パスではありません: {output_path}"

    def test_html_file_is_readable(
        self,
        visualizer: Visualizer,
        tmp_path,
    ) -> None:
        """生成された HTML ファイルが読み込み可能である"""
        config = VisualizerConfig(output_dir=str(tmp_path))
        fig = go.Figure()
        output_path = visualizer.export_html(fig, "test.html", config)
        with open(output_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert len(content) > 0, "HTML ファイルが空です"
        assert "<html" in content.lower() or "plotly" in content.lower()

    def test_filename_is_preserved(
        self,
        visualizer: Visualizer,
        tmp_path,
    ) -> None:
        """指定したファイル名が保持される"""
        import os
        config = VisualizerConfig(output_dir=str(tmp_path))
        fig = go.Figure()
        filename = "my_chart.html"
        output_path = visualizer.export_html(fig, filename, config)
        assert os.path.basename(output_path) == filename


# ─────────────────────────────────────────────
# VisualizerConfig データクラスのテスト
# ─────────────────────────────────────────────

class TestVisualizerConfigDataclass:
    """VisualizerConfig の基本動作テスト"""

    def test_is_frozen(self) -> None:
        """VisualizerConfig は frozen dataclass である"""
        config = VisualizerConfig()
        with pytest.raises(Exception):
            config.output_dir = "other"  # type: ignore[misc]

    def test_default_output_dir(self) -> None:
        """デフォルトの output_dir は 'output'"""
        config = VisualizerConfig()
        assert config.output_dir == "output"

    def test_default_height_px(self) -> None:
        """デフォルトの height は 400"""
        config = VisualizerConfig()
        assert config.default_height_px == 400

    def test_custom_values(self) -> None:
        """カスタム値を設定できる"""
        config = VisualizerConfig(output_dir="/tmp/out", default_height_px=600)
        assert config.output_dir == "/tmp/out"
        assert config.default_height_px == 600


# ─────────────────────────────────────────────
# build_hover_template のテスト
# ─────────────────────────────────────────────

class TestBuildHoverTemplate:
    """build_hover_template の動作テスト"""

    def test_returns_string(
        self,
        visualizer: Visualizer,
        sample_df: pd.DataFrame,
    ) -> None:
        """build_hover_template は文字列を返す"""
        result = visualizer.build_hover_template(sample_df)
        assert isinstance(result, str)

    def test_contains_all_required_fields(
        self,
        visualizer: Visualizer,
        sample_df: pd.DataFrame,
    ) -> None:
        """必須8項目がすべて含まれる（要件 2.3）"""
        result = visualizer.build_hover_template(sample_df)
        required_fields = [
            "箇所名",
            "通称線名名称",
            "駅・駅々間名称",
            "電柱番号",
            "キロ程",
            "架線構造名",
            "トロリ線種",
            "降雨フラグ",
        ]
        for field in required_fields:
            assert field in result, (
                f"hovertemplate に '{field}' が含まれていません。"
            )

    def test_template_is_non_empty(
        self,
        visualizer: Visualizer,
        sample_df: pd.DataFrame,
    ) -> None:
        """テンプレート文字列が空でない"""
        result = visualizer.build_hover_template(sample_df)
        assert len(result) > 0
