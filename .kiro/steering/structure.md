# Project Structure

## Organization Philosophy

パイプラインアーキテクチャに対応したレイヤー分割構成。
`src/` 配下にコンポーネント単位のモジュールを配置し、`app.py` がエントリポイントとしてパイプライン全体を統合する。

## Directory Patterns

### Application Entry Point
**Location**: `/app.py`
**Purpose**: Streamlit アプリのエントリポイント。サイドバーウィジェット定義とパイプライン呼び出しを担当
**Example**: `uv run streamlit run app.py` で起動

### Source Modules
**Location**: `/src/`
**Purpose**: パイプラインの各コンポーネントを1ファイル1モジュールで配置
**Pattern**: レイヤー別のコンポーネントがそれぞれ独立したモジュールとして存在
```
src/
  data_loader.py       # DataLayer: Excel読み込み・列検証・CH分割
  noise_filter.py      # ProcessingLayer: 移動中央値・SGフィルタ
  signal_analyzer.py   # ProcessingLayer: RMS・FFT・STFT
  anomaly_detector.py  # ProcessingLayer: Z-Score異常検知
  visualizer.py        # PresentationLayer: Plotlyグラフ生成
```

### Data Directory
**Location**: `/data/`
**Purpose**: 開発用サンプル Excel データの格納
**Example**: `data/20220916-koga-st-5cm-original-data.xlsx`

### Output Directory
**Location**: `/output/`
**Purpose**: `Visualizer.export_html` が生成する HTML ファイルの出力先
**Pattern**: ディレクトリ不存在時は自動作成

### Analysis Plans
**Location**: `/plan/`
**Purpose**: EDA 実行計画等の分析設計ドキュメント

## Naming Conventions

- **Files**: `snake_case.py`（例: `data_loader.py`, `noise_filter.py`）
- **Classes**: `PascalCase`（例: `DataLoader`, `NoiseFilter`, `AnomalyDetector`）
- **Dataclasses (frozen)**: `PascalCase` で名前が「Result」「Config」「Error」で終わるパターン
  - 結果値: `FilterResult`, `AnomalyResult`, `RMSResult`, `FFTResult`, `STFTResult`
  - 設定値: `FilterConfig`, `VisualizerConfig`, `ValidatedParams`
  - エラー値: `LoadError`, `InvalidWindowError`, `InvalidThresholdError`
- **Constants**: `UPPER_SNAKE_CASE`（例: `SAMPLE_SPACING_M`, `REQUIRED_COLUMNS`）
- **Functions/Methods**: `snake_case`（例: `validate_columns`, `compute_zscore`）

## Import Organization

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np
from scipy import signal
import plotly.graph_objects as go
from plotly.subplots import make_subplots
```

**Dependency Rule**: 上位レイヤーが下位レイヤーに依存する方向のみ許可
- `app.py` → `src/` の全モジュール
- `visualizer.py` → `noise_filter.py`, `anomaly_detector.py`, `signal_analyzer.py`（型参照のみ）
- `noise_filter.py` / `anomaly_detector.py` / `signal_analyzer.py` → 外部ライブラリのみ（相互依存なし）
- `data_loader.py` → pandas / openpyxl のみ

## Code Organization Principles

- **エラーは値として返す**: コンポーネントは例外を外部に送出せず、エラー値オブジェクトを返す
- **不変データクラス**: コンポーネント間の受け渡しオブジェクトはすべて `frozen=True` の dataclass
- **Visualizer は UI フレームワーク非依存**: `Visualizer` は `go.Figure` を返すのみ。Streamlit への依存は `app.py` に集約
- **パラメータ検証の集中管理**: ユーザ入力の検証はすべて `ParameterValidator` が担当し、他のコンポーネントは検証済み値を受け取る前提で動作する

---
_Document patterns, not file trees. New files following patterns shouldn't require updates_
