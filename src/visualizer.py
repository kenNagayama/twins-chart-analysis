"""PresentationLayer: Plotlyグラフ生成"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.anomaly_detector import AnomalyResult
from src.noise_filter import FilterResult
from src.signal_analyzer import FFTResult, RMSResult, STFTResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VisualizerConfig:
    """Visualizer の設定値オブジェクト。

    Attributes:
        output_dir: HTML 出力先ディレクトリ
        default_height_px: 1グラフあたりの高さ（ピクセル）
    """

    output_dir: str = "output"
    default_height_px: int = 400


class Visualizer:
    """全グラフのインタラクティブ可視化と出力を担当するクラス。

    - CH 別チャートでは hovertemplate に要件 2.3 の8項目を含める
    - 統合比較ビューは make_subplots(rows=3, cols=1, shared_xaxes=True) で構成する
    - HTML 出力は fig.write_html(output_path) で行う
    - グラフの軸ラベルは日本語を使用する
    - Streamlit に依存せず go.Figure を返すのみ（プレゼンテーション層の責務分離）

    Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 3.5, 4.7, 5.5, 6.1, 6.2, 6.3, 6.4, 6.5
    """

    def plot_channels(
        self,
        channel_data: dict[int, pd.DataFrame],
        config: VisualizerConfig,
    ) -> go.Figure:
        """CH 1〜4 の摩耗チャートを生成する。

        各 CH を個別サブプロットに表示する。
        hovertemplate に要件 2.3 で定義した8項目を含める。

        Args:
            channel_data: {ch番号: DataFrame} の辞書（CH 1〜4 対応）
            config: Visualizer 設定

        Returns:
            全 CH の摩耗チャートを含む go.Figure

        Requirements: 2.1, 2.2, 2.3, 2.4, 2.5
        """
        ch_numbers = sorted(channel_data.keys())
        n_chs = len(ch_numbers)

        if n_chs == 0:
            # データなしの場合は空の Figure を返す
            return go.Figure()

        # CH 数に応じてサブプロットレイアウトを決定
        fig = make_subplots(
            rows=n_chs,
            cols=1,
            shared_xaxes=True,
            subplot_titles=[f"CH{ch}" for ch in ch_numbers],
            vertical_spacing=0.04,
        )

        for row_idx, ch in enumerate(ch_numbers, start=1):
            df = channel_data[ch]

            # ホバーテンプレートを構築
            hover_template = self.build_hover_template(df)

            # カスタムデータ（ホバー用の追加列）を設定
            custom_data = _build_custom_data(df)

            trace = go.Scatter(
                x=df["キロ程"].values,
                y=df["摩耗_測定値"].values,
                mode="lines",
                name=f"CH{ch}",
                hovertemplate=hover_template,
                customdata=custom_data,
            )
            fig.add_trace(trace, row=row_idx, col=1)

        # レイアウト設定（日本語ラベル・フォント）
        fig.update_layout(
            title="CH別 トロリ線摩耗チャート",
            font=dict(family="Meiryo, sans-serif"),
            height=config.default_height_px * max(n_chs, 1),
        )

        # Y軸ラベルを設定
        for row_idx in range(1, n_chs + 1):
            fig.update_yaxes(title_text="残存直径 (mm)", row=row_idx, col=1)

        # 最下段の X 軸ラベルを設定
        fig.update_xaxes(title_text="キロ程 (m)", row=n_chs, col=1)

        return fig

    def plot_filter_comparison(
        self,
        filter_result: FilterResult,
        kilometric_series: pd.Series,
        config: VisualizerConfig,
    ) -> go.Figure:
        """フィルタ前後の波形を重ねた比較グラフを生成する。

        横軸にキロ程、縦軸に残存直径を使用し、2系列を色分けする。

        Args:
            filter_result: FilterResult（元データ・フィルタ後データ・設定を保持）
            kilometric_series: キロ程値の Series
            config: Visualizer 設定

        Returns:
            フィルタ前後の比較グラフを含む go.Figure

        Requirements: 4.7
        """
        fig = go.Figure()

        # フィルタ前の波形（元データ）
        fig.add_trace(
            go.Scatter(
                x=kilometric_series.values,
                y=filter_result.original.values,
                mode="lines",
                name="フィルタ前（元データ）",
                line=dict(color="steelblue", width=1),
                hovertemplate=(
                    "キロ程: %{x:.3f} m<br>"
                    "残存直径（元）: %{y:.3f} mm<extra></extra>"
                ),
            )
        )

        # フィルタ後の波形
        fig.add_trace(
            go.Scatter(
                x=kilometric_series.values,
                y=filter_result.filtered.values,
                mode="lines",
                name="フィルタ後",
                line=dict(color="orangered", width=1.5),
                hovertemplate=(
                    "キロ程: %{x:.3f} m<br>"
                    "残存直径（フィルタ後）: %{y:.3f} mm<extra></extra>"
                ),
            )
        )

        # レイアウト設定
        fig.update_layout(
            title="フィルタ前後 比較グラフ",
            xaxis_title="キロ程 (m)",
            yaxis_title="残存直径 (mm)",
            font=dict(family="Meiryo, sans-serif"),
            height=config.default_height_px,
        )

        return fig

    def plot_analysis_results(
        self,
        rms_result: RMSResult,
        fft_result: FFTResult,
        stft_result: STFTResult,
        config: VisualizerConfig,
    ) -> go.Figure:
        """RMS・FFT・STFT の結果をグラフ化する。

        RMS 波形・FFT 振幅スペクトル・STFT スペクトログラムをそれぞれ個別サブプロットで描画する。
        各グラフに適切な日本語軸ラベルを設定する。

        Args:
            rms_result: RMSResult（rms_series を含む）
            fft_result: FFTResult（frequencies, amplitudes を含む）
            stft_result: STFTResult（frequencies, positions, spectrogram を含む）
            config: Visualizer 設定

        Returns:
            RMS・FFT・STFT の信号解析結果グラフを含む go.Figure

        Requirements: 3.5
        """
        fig = make_subplots(
            rows=3,
            cols=1,
            subplot_titles=["RMS 波形", "FFT 振幅スペクトル", "STFT スペクトログラム"],
            vertical_spacing=0.08,
        )

        # サブプロット行1: RMS 波形
        rms_series = rms_result.rms_series
        fig.add_trace(
            go.Scatter(
                x=list(range(len(rms_series))),
                y=rms_series.values,
                mode="lines",
                name="RMS",
                line=dict(color="steelblue"),
                hovertemplate="インデックス: %{x}<br>RMS: %{y:.4f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        # サブプロット行2: FFT 振幅スペクトル
        fig.add_trace(
            go.Scatter(
                x=fft_result.frequencies,
                y=fft_result.amplitudes,
                mode="lines",
                name="FFT",
                line=dict(color="green"),
                hovertemplate="周波数: %{x:.3f} m⁻¹<br>振幅: %{y:.4f}<extra></extra>",
            ),
            row=2,
            col=1,
        )

        # サブプロット行3: STFT スペクトログラム（ヒートマップ）
        fig.add_trace(
            go.Heatmap(
                x=stft_result.positions,
                y=stft_result.frequencies,
                z=stft_result.spectrogram,
                name="STFT",
                colorscale="Viridis",
                hovertemplate=(
                    "位置: %{x:.3f} m<br>"
                    "周波数: %{y:.3f} m⁻¹<br>"
                    "振幅: %{z:.4f}<extra></extra>"
                ),
            ),
            row=3,
            col=1,
        )

        # レイアウト設定（日本語ラベル）
        fig.update_layout(
            title="信号解析結果（RMS / FFT / STFT）",
            font=dict(family="Meiryo, sans-serif"),
            height=config.default_height_px * 3,
        )

        # 軸ラベル
        fig.update_yaxes(title_text="RMS 値", row=1, col=1)
        fig.update_xaxes(title_text="インデックス", row=1, col=1)
        fig.update_yaxes(title_text="振幅", row=2, col=1)
        fig.update_xaxes(title_text="周波数 (m⁻¹)", row=2, col=1)
        fig.update_yaxes(title_text="周波数 (m⁻¹)", row=3, col=1)
        fig.update_xaxes(title_text="位置 (m)", row=3, col=1)

        return fig

    def plot_comparison_view(
        self,
        raw_series: pd.Series,
        filtered_series: pd.Series,
        anomaly_result: AnomalyResult,
        kilometric_series: pd.Series,
        config: VisualizerConfig,
    ) -> go.Figure:
        """生波形・フィルタ後波形・Z-Scoreの3段連動サブプロットを生成する。

        make_subplots(rows=3, cols=1, shared_xaxes=True) を使って
        3グラフの横軸（キロ程）を連動させる。
        Z-Score プロット上に閾値ラインを破線で表示する。

        Args:
            raw_series: 生データの摩耗値 Series
            filtered_series: フィルタ後の摩耗値 Series
            anomaly_result: AnomalyResult（Z-Score・異常点を含む）
            kilometric_series: キロ程値の Series
            config: Visualizer 設定

        Returns:
            3段連動サブプロットを含む go.Figure

        Requirements: 6.1, 6.2, 6.3, 6.4
        """
        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            subplot_titles=["生波形", "フィルタ後波形", "Z-Score"],
            vertical_spacing=0.04,
        )

        kilo_vals = kilometric_series.values

        # 行1: 生波形
        fig.add_trace(
            go.Scatter(
                x=kilo_vals,
                y=raw_series.values,
                mode="lines",
                name="生波形",
                line=dict(color="steelblue", width=1),
                hovertemplate=(
                    "キロ程: %{x:.3f} m<br>"
                    "測定値: %{y:.3f} mm<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

        # 行2: フィルタ後波形
        fig.add_trace(
            go.Scatter(
                x=kilo_vals,
                y=filtered_series.values,
                mode="lines",
                name="フィルタ後",
                line=dict(color="orangered", width=1.5),
                hovertemplate=(
                    "キロ程: %{x:.3f} m<br>"
                    "測定値（フィルタ後）: %{y:.3f} mm<extra></extra>"
                ),
            ),
            row=2,
            col=1,
        )

        # 行3: Z-Score プロット
        zscore_vals = anomaly_result.zscore_series.values
        fig.add_trace(
            go.Scatter(
                x=kilo_vals,
                y=zscore_vals,
                mode="lines",
                name="Z-Score",
                line=dict(color="purple", width=1),
                hovertemplate=(
                    "キロ程: %{x:.3f} m<br>"
                    "Z-Score: %{y:.3f}<extra></extra>"
                ),
            ),
            row=3,
            col=1,
        )

        # 閾値ライン（破線）を Z-Score プロットに追加（要件 6.3）
        threshold = anomaly_result.threshold
        x_range = [float(kilo_vals[0]), float(kilo_vals[-1])] if len(kilo_vals) > 0 else [0, 1]

        # 正の閾値ライン
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=[threshold, threshold],
                mode="lines",
                name=f"閾値 +{threshold}",
                line=dict(color="red", dash="dash", width=1.5),
                hovertemplate=f"閾値: +{threshold}<extra></extra>",
            ),
            row=3,
            col=1,
        )

        # 負の閾値ライン
        fig.add_trace(
            go.Scatter(
                x=x_range,
                y=[-threshold, -threshold],
                mode="lines",
                name=f"閾値 -{threshold}",
                line=dict(color="red", dash="dash", width=1.5),
                hovertemplate=f"閾値: -{threshold}<extra></extra>",
            ),
            row=3,
            col=1,
        )

        # レイアウト設定
        fig.update_layout(
            title="統合比較ビュー（生波形 / フィルタ後 / Z-Score）",
            font=dict(family="Meiryo, sans-serif"),
            height=config.default_height_px * 3,
        )

        # 軸ラベル
        fig.update_yaxes(title_text="残存直径 (mm)", row=1, col=1)
        fig.update_yaxes(title_text="残存直径 (mm)", row=2, col=1)
        fig.update_yaxes(title_text="Z-Score", row=3, col=1)
        fig.update_xaxes(title_text="キロ程 (m)", row=3, col=1)

        return fig

    def build_hover_template(self, df: pd.DataFrame) -> str:
        """要件 2.3 で定義された8項目のホバーテンプレート文字列を構築する。

        8項目:
            1. 箇所名（日本語テキスト）
            2. 通称線名名称（日本語テキスト）
            3. 駅・駅々間名称（日本語テキスト）
            4. 電柱番号（文字列）
            5. キロ程（小数第3位まで表示）
            6. 架線構造名（テキスト）
            7. トロリ線種（テキスト）
            8. 降雨フラグ（文字列）

        customdata の列順は _build_custom_data() に準じる。

        Args:
            df: 必須列を含む DataFrame

        Returns:
            Plotly の hovertemplate 文字列

        Requirements: 2.3
        """
        # customdata の列インデックス（_build_custom_data() と同順）
        # 0: 箇所名, 1: 通称線名名称, 2: 駅・駅々間名称, 3: 電柱番号
        # 4: 架線構造名, 5: トロリ線種, 6: 降雨フラグ
        template = (
            "箇所名: %{customdata[0]}<br>"
            "通称線名名称: %{customdata[1]}<br>"
            "駅・駅々間名称: %{customdata[2]}<br>"
            "電柱番号: %{customdata[3]}<br>"
            "キロ程: %{x:.3f} m<br>"
            "架線構造名: %{customdata[4]}<br>"
            "トロリ線種: %{customdata[5]}<br>"
            "降雨フラグ: %{customdata[6]}"
            "<extra></extra>"
        )
        return template

    def plot_anomaly_overlay(
        self,
        series: pd.Series,
        kilometric_series: pd.Series,
        anomaly_result: "AnomalyResult",
        config: VisualizerConfig,
    ) -> go.Figure:
        """摩耗波形に異常点マーカーをオーバーレイしたグラフを生成する。

        AnomalyDetector が返した異常点インデックスを使って、グラフ上の
        該当データポイントをマーカーでハイライト表示する。
        ハイライトには閾値超過点のキロ程値を示す。

        Args:
            series: 摩耗値 Series（フィルタリング済みまたは生データ）
            kilometric_series: キロ程値の Series
            anomaly_result: AnomalyResult（異常点インデックス・キロ程を含む）
            config: Visualizer 設定

        Returns:
            摩耗波形 + 異常点マーカーを含む go.Figure

        Requirements: 5.5
        """
        fig = go.Figure()

        # ベース波形トレース
        fig.add_trace(
            go.Scatter(
                x=kilometric_series.values,
                y=series.values,
                mode="lines",
                name="摩耗波形",
                line=dict(color="steelblue", width=1),
                hovertemplate=(
                    "キロ程: %{x:.3f} m<br>"
                    "残存直径: %{y:.3f} mm<extra></extra>"
                ),
            )
        )

        # 異常点マーカートレース（異常点が存在する場合のみ追加）
        anomaly_idx = anomaly_result.anomaly_indices
        if len(anomaly_idx) > 0:
            # 異常点の摩耗値とキロ程を取得
            anomaly_wear = series.loc[anomaly_idx]
            anomaly_kilo = kilometric_series.reindex(anomaly_idx)

            fig.add_trace(
                go.Scatter(
                    x=anomaly_kilo.values,
                    y=anomaly_wear.values,
                    mode="markers",
                    name=f"異常点（閾値 ±{anomaly_result.threshold}）",
                    marker=dict(
                        color="red",
                        size=8,
                        symbol="circle-open",
                        line=dict(width=2),
                    ),
                    hovertemplate=(
                        "【異常点】<br>"
                        "キロ程: %{x:.3f} m<br>"
                        "残存直径: %{y:.3f} mm<extra></extra>"
                    ),
                )
            )

        # レイアウト設定
        fig.update_layout(
            title=f"摩耗波形 異常点ハイライト（Z-Score 閾値 ±{anomaly_result.threshold}）",
            xaxis_title="キロ程 (m)",
            yaxis_title="残存直径 (mm)",
            font=dict(family="Meiryo, sans-serif"),
            height=config.default_height_px,
        )

        return fig

    def export_html(
        self,
        fig: go.Figure,
        filename: str,
        config: VisualizerConfig,
    ) -> str:
        """図を HTML ファイルとして保存し、出力パスを返す。

        出力先ディレクトリが存在しない場合は自動作成する。

        Args:
            fig: 出力する Plotly Figure
            filename: 出力ファイル名（例: "chart.html"）
            config: Visualizer 設定（output_dir を使用）

        Returns:
            書き込んだファイルの絶対パス

        Requirements: 6.5
        """
        output_dir = Path(config.output_dir)

        # ディレクトリが存在しない場合は自動作成
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / filename
        fig.write_html(str(output_path))

        return str(output_path.resolve())


# ─────────────────────────────────────────────
# モジュール内ユーティリティ関数
# ─────────────────────────────────────────────

def _build_custom_data(df: pd.DataFrame):
    """hovertemplate 用の customdata 配列を構築する。

    列順:
        0: 箇所名
        1: 通称線名名称
        2: 駅・駅々間名称
        3: 電柱番号
        4: 架線構造名
        5: トロリ線種
        6: 降雨フラグ

    Args:
        df: 必須列を含む DataFrame

    Returns:
        shape (n, 7) の配列（Plotly customdata 向け）
    """
    columns = [
        "箇所名",
        "通称線名名称",
        "駅・駅々間名称",
        "電柱番号",
        "架線構造名",
        "トロリ線種",
        "降雨フラグ",
    ]
    return np.column_stack([df[col].values for col in columns])
