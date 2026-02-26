# Technology Stack

## Architecture

パイプラインアーキテクチャを採用。各コンポーネントが単一の変換責任を持ち、前段の出力を後段の入力として渡す直列構成。
DataLayer → ProcessingLayer → PresentationLayer の3層で責務を分離し、各コンポーネントが独立してテスト可能。

## Core Technologies

- **Language**: Python 3.10+
- **Framework**: Streamlit（Web UI）、Plotly（インタラクティブ可視化）
- **Runtime**: SageMaker Studio 環境
- **Package Manager**: uv（`pyproject.toml` で依存管理、`uv run streamlit run app.py` で起動）

## Key Libraries

- **pandas 2.x**: DataFrame によるデータ管理、ローリング統計（`rolling().median()`, `rolling().mean()`, `rolling().std()`）
- **NumPy 1.26+**: FFT 計算（`numpy.fft.rfft`, `rfftfreq`）、配列演算
- **SciPy 1.17.1**: `savgol_filter`（Savitzky-Golay フィルタ）、`scipy.signal.stft`（短時間フーリエ変換）
- **Plotly 6.5.0**: インタラクティブチャート、`make_subplots(shared_xaxes=True)` による連動サブプロット、HTML 出力
- **openpyxl**: Excel ファイル読み込みバックエンド（pandas 経由）

## Development Standards

### Type Safety
- Python の型ヒント（`from __future__ import annotations`）を使用
- 不変値オブジェクトは `@dataclass(frozen=True)` で定義する
- 戻り値の Union 型（例: `pd.DataFrame | LoadError`）でエラーを値として返し、例外を外部に送出しない設計

### Code Quality
- エラーハンドリング: Fail Fast 方針。エラー値オブジェクト（`LoadError`, `InvalidWindowError`）を返すパターンを採用
- UI エラー表示: Streamlit の `st.error()` / `st.warning()` を使用
- ログ: `logging.warning` でターミナルに出力

### Testing
- `pytest` によるユニットテスト（DataLoader, ParameterValidator, NoiseFilter, AnomalyDetector）
- 小規模 fixture DataFrame を用いたパイプライン統合テスト
- E2E テスト: `uv run streamlit run app.py --headless` での起動確認

## Development Environment

### Required Tools
- Python 3.10+
- uv（パッケージ管理・実行）

### Common Commands
```bash
# Dev: uv run streamlit run app.py
# Test: uv run pytest
# Sync deps: uv sync
```

## Key Technical Decisions

- **エラーを例外でなく値で返す**: `DataLoader` と `ParameterValidator` はエラーを例外送出ではなく値オブジェクトとして返す。Streamlit のリラン特性に適合し、UI 上のエラー表示と組み合わせやすい
- **Visualizer は Streamlit に依存しない**: `Visualizer` は `go.Figure` を返すのみ。`StreamlitApp` が `st.plotly_chart` に渡す。責務分離により単体テスト可能
- **5cm ピッチの定数化**: `SignalAnalyzer.SAMPLE_SPACING_M = 0.05`（5cm = 0.05m）として定数定義し、周波数軸計算を統一する
- **Savitzky-Golay の奇数制約**: SG フィルタのウィンドウ幅は奇数必須。`ParameterValidator.ensure_odd_window` で偶数入力を +1 補正して保証する

---
_Document standards and patterns, not every dependency_
