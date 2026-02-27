"""Streamlit アプリのエントリポイント

タスク 8: Streamlitアプリの統合実装
  - 8.1: サイドバーパラメータウィジェットの実装
  - 8.2: メイン画面へのグラフ描画統合
  - 8.3: 異常検知結果のグラフハイライト統合
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.anomaly_detector import AnomalyDetector
from src.data_loader import DataLoader, LoadError
from src.noise_filter import FilterConfig, NoiseFilter
from src.parameter_validator import (
    InvalidThresholdError,
    InvalidWindowError,
    ParameterValidator,
)
from src.signal_analyzer import SignalAnalyzer
from src.visualizer import Visualizer, VisualizerConfig

# ─────────────────────────────────────────────
# 定数
# ─────────────────────────────────────────────

# デフォルトデータファイルパス
DEFAULT_DATA_PATH = (
    Path(__file__).parent / "data" / "20220916-koga-st-5cm-original-data.xlsx"
)

# アプリタイトル
APP_TITLE = "トロリ線摩耗検測データ 探索的データ分析（EDA）ツール"

# ─────────────────────────────────────────────
# ページ設定
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="トロリ線摩耗 EDA ツール",
    page_icon="📊",
    layout="wide",
)

st.title(APP_TITLE)


# ─────────────────────────────────────────────
# データ読み込み
# ─────────────────────────────────────────────

@st.cache_data
def load_data(file_path: str):
    """Excelファイルを読み込み DataFrame を返す（キャッシュ付き）。"""
    loader = DataLoader()
    return loader.load(file_path)


# データの読み込み
data_path = str(DEFAULT_DATA_PATH)
df_or_error = load_data(data_path)

if isinstance(df_or_error, LoadError):
    if df_or_error.kind == "file_not_found":
        st.error(
            f"データファイルが見つかりません: {data_path}\n\n"
            "正しいパスを確認してください。"
        )
    elif df_or_error.kind == "missing_columns":
        cols = ", ".join(df_or_error.missing_columns)
        st.error(
            f"必須列が不足しています: {cols}\n\n"
            "データファイルの列構成を確認してください。"
        )
    else:
        st.error(f"データ読み込みエラー: {df_or_error.message}")
    st.stop()

# 読み込み成功 - df_or_error は pd.DataFrame
df = df_or_error

# ─────────────────────────────────────────────
# タスク 8.1: サイドバーパラメータウィジェットの実装
# ─────────────────────────────────────────────

st.sidebar.header("分析パラメータ")

# データ全体の長さ（スライダーの最大値に使用）
total_data_length = len(df)

# ウィンドウ幅スライダー
window_size_input: int = st.sidebar.slider(
    "ウィンドウ幅",
    min_value=1,
    max_value=min(total_data_length, 500),
    value=min(51, total_data_length),
    step=1,
    help="移動中央値・Savitzky-Golay・RMS・Z-Score算出に使用するウィンドウ幅",
)

# Z-Score 閾値入力
zscore_threshold_input: float = st.sidebar.number_input(
    "Z-Score 閾値",
    min_value=0.1,
    max_value=10.0,
    value=2.5,
    step=0.1,
    format="%.1f",
    help="この値を超えた Z-Score を異常と判定します（正の実数）",
)

# 移動中央値フィルタ ON/OFF トグル
median_enabled: bool = st.sidebar.toggle(
    "移動中央値フィルタ",
    value=True,
    help="移動中央値フィルタを有効にするとスパイクノイズを除去します",
)

# Savitzky-Golay フィルタ ON/OFF トグル
savgol_enabled: bool = st.sidebar.toggle(
    "Savitzky-Golay フィルタ",
    value=True,
    help="Savitzky-Golay フィルタで滑らかな曲線を生成します",
)

# ─────────────────────────────────────────────
# パラメータバリデーション
# ─────────────────────────────────────────────

validator = ParameterValidator()

# ウィンドウ幅バリデーション
window_validation = validator.validate_window(window_size_input, total_data_length)
if isinstance(window_validation, InvalidWindowError):
    st.error(f"ウィンドウ幅エラー: {window_validation.message}")
    st.stop()

window_size: int = window_validation

# Savitzky-Golay 向けに奇数補正
odd_window = validator.ensure_odd_window(window_size)

# Z-Score 閾値バリデーション
threshold_validation = validator.validate_threshold(zscore_threshold_input)
if isinstance(threshold_validation, InvalidThresholdError):
    st.error(f"Z-Score 閾値エラー: {threshold_validation.message}")
    st.stop()

zscore_threshold: float = threshold_validation

# ─────────────────────────────────────────────
# CH別データの準備
# ─────────────────────────────────────────────

loader = DataLoader()
channel_data: dict[int, object] = {}
for ch in range(1, 5):
    df_ch = loader.get_channel_group(df, ch)
    if len(df_ch) > 0:
        channel_data[ch] = df_ch.reset_index(drop=True)

if not channel_data:
    st.warning("CH 1〜4 のデータが見つかりませんでした。")
    st.stop()

# 最初に有効な CH のデータを代表データとして使用
first_ch = min(channel_data.keys())
representative_df = channel_data[first_ch]
wear_series = representative_df["摩耗_測定値"]
kilo_series = representative_df["キロ程"]

# ─────────────────────────────────────────────
# フィルタリング処理
# ─────────────────────────────────────────────

filter_config = FilterConfig(
    median_enabled=median_enabled,
    median_window=odd_window,
    savgol_enabled=savgol_enabled,
    savgol_window=odd_window,
    savgol_polyorder=2,
)

noise_filter = NoiseFilter()
filter_result = noise_filter.apply(wear_series, filter_config)

# ─────────────────────────────────────────────
# 異常検知処理
# ─────────────────────────────────────────────

detector = AnomalyDetector()
anomaly_result = detector.detect(
    filter_result.filtered,
    window_size=window_size,
    threshold=zscore_threshold,
    position_series=kilo_series,
)

# ─────────────────────────────────────────────
# 信号解析処理
# ─────────────────────────────────────────────

analyzer = SignalAnalyzer()
rms_result = analyzer.compute_rms(filter_result.filtered, window_size)
fft_result = analyzer.compute_fft(filter_result.filtered, window_size)
stft_result = analyzer.compute_stft(filter_result.filtered, window_size)

# ─────────────────────────────────────────────
# タスク 8.2: メイン画面へのグラフ描画統合
# ─────────────────────────────────────────────

visualizer = Visualizer()
viz_config = VisualizerConfig(output_dir="output", default_height_px=400)

# ─ セクション1: CH別摩耗チャート ──────────────────

st.header("CH別 摩耗チャート")
st.markdown("各測定チャンネル（CH 1〜4）の摩耗分布をインタラクティブに表示します。")

ch_fig = visualizer.plot_channels(channel_data, viz_config)
st.plotly_chart(ch_fig, use_container_width=True)

# ─ セクション2: フィルタ前後比較グラフ ─────────────

st.header("フィルタ前後 比較グラフ")
st.markdown(
    "生波形とフィルタ後波形を重ねて比較します。"
    f"（移動中央値: {'ON' if median_enabled else 'OFF'} / "
    f"Savitzky-Golay: {'ON' if savgol_enabled else 'OFF'}）"
)

filter_comparison_fig = visualizer.plot_filter_comparison(
    filter_result, kilo_series, viz_config
)
st.plotly_chart(filter_comparison_fig, use_container_width=True)

# ─ セクション3: 信号解析グラフ ────────────────────

st.header("信号解析結果（RMS / FFT / STFT）")
st.markdown(
    f"ウィンドウ幅 {window_size} でのスライドウィンドウ RMS・FFT 振幅スペクトル・STFT スペクトログラムを表示します。"
)

analysis_fig = visualizer.plot_analysis_results(rms_result, fft_result, stft_result, viz_config)
st.plotly_chart(analysis_fig, use_container_width=True)

# ─ セクション4: 統合比較ビュー（生波形 / フィルタ後 / Z-Score）──

st.header("統合比較ビュー")
st.markdown(
    "生波形・フィルタ後波形・Z-Score の3段連動グラフです。"
    "横軸（キロ程）はズーム・パン操作が同期します。"
)

comparison_fig = visualizer.plot_comparison_view(
    raw_series=wear_series,
    filtered_series=filter_result.filtered,
    anomaly_result=anomaly_result,
    kilometric_series=kilo_series,
    config=viz_config,
)
st.plotly_chart(comparison_fig, use_container_width=True)

# 異常点の件数を表示
n_anomalies = len(anomaly_result.anomaly_indices)
if n_anomalies > 0:
    st.warning(
        f"Z-Score 閾値 ±{zscore_threshold} を超えた異常点が "
        f"**{n_anomalies} 件** 検出されました。"
    )
else:
    st.info(f"Z-Score 閾値 ±{zscore_threshold} を超えた異常点は検出されませんでした。")

# ─────────────────────────────────────────────
# タスク 8.3: 異常検知結果のグラフハイライト統合
# ─────────────────────────────────────────────

st.header("異常点ハイライト表示")
st.markdown(
    "フィルタ後波形上に異常点（Z-Score 閾値超過）をマーカーでハイライト表示します。"
    "各マーカーはキロ程値を示します。"
)

# AnomalyDetector が返した異常点インデックスを使ってハイライトグラフを生成
anomaly_overlay_fig = visualizer.plot_anomaly_overlay(
    filter_result.filtered,
    kilo_series,
    anomaly_result,
    viz_config,
)
st.plotly_chart(anomaly_overlay_fig, use_container_width=True)

# 異常点の詳細テーブル表示（異常点がある場合のみ）
if n_anomalies > 0:
    st.subheader("異常点一覧")
    anomaly_positions = anomaly_result.anomaly_positions
    anomaly_idx = anomaly_result.anomaly_indices

    # 異常点のデータを抽出してテーブル表示
    anomaly_details = representative_df.loc[anomaly_idx][
        ["キロ程", "摩耗_測定値", "箇所名", "通称線名名称", "電柱番号"]
    ].copy()
    anomaly_details["Z-Score"] = anomaly_result.zscore_series.loc[anomaly_idx].values
    anomaly_details = anomaly_details.reset_index(drop=True)
    st.dataframe(anomaly_details, use_container_width=True)

# ─────────────────────────────────────────────
# HTML 出力ボタン（タスク 8.2）
# ─────────────────────────────────────────────

st.divider()
st.header("HTML 出力")
st.markdown("分析結果を HTML ファイルとして `output/` ディレクトリに保存します。")

col1, col2 = st.columns(2)

with col1:
    if st.button("統合比較ビューを HTML 出力", type="primary"):
        output_path = visualizer.export_html(comparison_fig, "comparison_view.html", viz_config)
        st.success(f"HTML ファイルを保存しました: `{output_path}`")

with col2:
    if st.button("異常点ハイライトを HTML 出力"):
        output_path = visualizer.export_html(
            anomaly_overlay_fig, "anomaly_overlay.html", viz_config
        )
        st.success(f"HTML ファイルを保存しました: `{output_path}`")
