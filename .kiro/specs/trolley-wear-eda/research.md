# リサーチ & 設計判断

---
**目的**: 技術設計を裏付ける調査結果・アーキテクチャ検討・判断根拠を記録する。

**使用方法**:
- ディスカバリフェーズ中の調査活動と結果をログとして残す。
- `design.md` に収めるには詳細すぎる設計判断のトレードオフを文書化する。
- 将来の監査や再利用のための参照・エビデンスを提供する。
---

## Summary

- **Feature**: `trolley-wear-eda`
- **Discovery Scope**: 新規フィーチャー（グリーンフィールド）— 既存コードベースに同等の分析ツールは存在しない
- **Key Findings**:
  - Plotly 6.5.0 の `make_subplots(shared_xaxes=True)` がリンクドサブプロット要件を直接充足する
  - SciPy 1.17.1 の `scipy.signal.savgol_filter` / `scipy.signal.stft` が要件 3〜4 の信号処理を網羅する
  - パイプライン層（DataLoader → Preprocessor → Analyzer → Visualizer）への分離が各要件を独立してテスト可能にする

---

## Research Log

### トピック 1: インタラクティブ可視化ライブラリの選定

- **Context**: 要件 2・6 でホバーツールチップ、ズーム/パン、連動軸、HTML 出力が求められる。
- **Sources Consulted**:
  - https://plotly.com/python/
  - https://plotly.com/python/subplots/
- **Findings**:
  - Plotly 6.5.0（2026年2月時点の最新安定版）
  - `make_subplots(rows=3, cols=1, shared_xaxes=True)` で縦3段・X軸連動のサブプロットを構築できる
  - `fig.write_html("output.html")` で単ファイル HTML 出力が可能（要件 6.5 充足）
  - `go.Scatter` の `hovertemplate` パラメータでツールチップ項目を自由に定義できる
  - Bokeh も候補に挙がるが、サブプロット連動と HTML 出力の簡便さで Plotly が優位
- **Implications**: フロントエンド層は Plotly のみに統一。Jupyter Notebook 上では `fig.show()`、スクリプト実行時は `write_html` を使い分けるファサードを設計する。

### トピック 2: 信号処理ライブラリ（SciPy）

- **Context**: 要件 3 で RMS・FFT・STFT、要件 4 で Savitzky-Golay フィルタが必要。
- **Sources Consulted**:
  - https://docs.scipy.org/doc/scipy/reference/signal.html
  - https://scipy.org/（バージョン確認）
- **Findings**:
  - SciPy 1.17.1（2026年2月22日リリース）
  - `scipy.signal.savgol_filter(x, window_length, polyorder)` — Savitzky-Golay フィルタ
  - `scipy.signal.stft(x, fs, window, nperseg, noverlap)` — STFT（レガシー関数だが安定）
  - `scipy.signal.ShortTimeFFT` — 新しい STFT クラス（同一長配列の反復計算に最適化）
  - FFT は `numpy.fft.rfft` / `rfftfreq` で実数入力向けに最適化された計算が可能
  - RMS はスライドウィンドウに対して `pandas.DataFrame.rolling(window).apply(rms_fn)` で実装可能
- **Implications**: 信号処理は `SignalAnalyzer` コンポーネントが `scipy.signal` + `numpy.fft` に依存する。STFT は `scipy.signal.stft` を使用し、将来的に `ShortTimeFFT` へ移行可能な抽象化を設ける。

### トピック 3: Pandas ローリング統計

- **Context**: 要件 4 の移動中央値フィルタ、要件 5 の移動平均・Z-Score 計算に使用。
- **Sources Consulted**:
  - https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rolling.html
- **Findings**:
  - `DataFrame.rolling(window, min_periods=1, center=True).median()` で移動中央値フィルタを実装
  - `DataFrame.rolling(window, min_periods=1).mean()` で移動平均
  - `min_periods` を `1` に設定することで端部（データ数不足区間）を NaN にせず扱える（ただし要件 5.6 では NaN での処理が明示されているため、Z-Score 計算側では `min_periods=window` とし端部を意図的に NaN とする）
- **Implications**: `NoiseFilter` と `AnomalyDetector` は pandas の rolling API をラップするシンプルな実装になる。

### トピック 4: 既存コードベース分析

- **Context**: 既存の Python スクリプト・Notebook が存在するか確認。
- **Findings**:
  - プロジェクトルートに Python ソースファイル・Jupyter Notebook は存在しない（グリーンフィールド）
  - `.claude/skills/` 配下にスキャンスクリプトがあるが、本フィーチャーとは無関係
  - `data/20220916-koga-st-5cm-original-data.xlsx` が参照されているが、実ファイルの存在は実装時に確認
- **Implications**: フルスクラッチで設計する。既存コードとの互換性考慮は不要。

---

## Architecture Pattern Evaluation

| オプション | 説明 | 強み | リスク/制限 | 備考 |
|---|---|---|---|---|
| **Pipeline（採用）** | DataLoader → Preprocessor → Analyzer → Visualizer の直列パイプライン | 各層が独立、テスト容易、要件ごとに責務が明確 | 状態管理が複雑化する場合あり | EDA ユースケースに最適マッチ |
| Monolithic Script | 単一スクリプトに全処理を記述 | 実装が速い | テスト困難、再利用不可、拡張性なし | 要件7（ファイルアップロード拡張）に対応不可 |
| Class-based OOP | 単一クラスに全メソッドを集約 | オブジェクト指向で整理 | 単一責任原則違反、テスト粒度が粗い | Pipeline よりも依存が密になる |

---

## Design Decisions

### Decision: `Jupyter Notebook + Python モジュール併用構成`

- **Context**: 要件 6.5 で HTML 出力とインタラクティブ Notebook 出力の両方が求められる。
- **Alternatives Considered**:
  1. Jupyter Notebook のみ — セルで全処理を記述
  2. Python スクリプトのみ — CLI でパラメータ指定して HTML 出力
  3. **Notebook + 共通モジュール（採用）** — 処理ロジックを Python モジュール (`src/`) に分離し、Notebook から呼び出す
- **Selected Approach**: `src/` ディレクトリに分析ロジック（`data_loader.py`, `noise_filter.py`, `signal_analyzer.py`, `anomaly_detector.py`, `visualizer.py`）を配置し、`notebook/eda.ipynb` からこれらを `import` して利用する。
- **Rationale**: Notebook 単体ではユニットテストが困難であり、要件 7（ファイルアップロード機能への拡張）で共通モジュールが再利用できる。
- **Trade-offs**: 初期実装コストがやや増えるが、保守性・テスト性・拡張性が大幅に向上する。
- **Follow-up**: `requirements.txt` のバージョンピン留めを実装時に確認すること。

### Decision: `スライドウィンドウの端部処理`

- **Context**: 要件 3.6 でウィンドウ幅バリデーション、要件 5.6 でデータ点不足時の NaN 処理が求められる。
- **Alternatives Considered**:
  1. `min_periods=1` — 端部も計算するが不完全なウィンドウの結果が混入する
  2. `min_periods=window_size`（採用） — ウィンドウが満たない区間は NaN
- **Selected Approach**: Z-Score 計算（要件 5）では `min_periods=window_size` とし端部を NaN とする。移動中央値フィルタ（要件 4）では `min_periods=1` を使用してフィルタの連続性を確保する。
- **Rationale**: Z-Score の統計的信頼性を保つ一方、フィルタリングの波形を途切れさせない。
- **Trade-offs**: Z-Score グラフで端部に NaN 区間が生じるが、要件 5.6 で明示的に仕様化されている。

### Decision: `FFT 実装方針 — numpy.fft.rfft の採用`

- **Context**: 要件 3.3 でスライドウィンドウ FFT が求められる。摩耗データは実数値のみ。
- **Alternatives Considered**:
  1. `numpy.fft.fft` — 複素入力汎用版
  2. `numpy.fft.rfft`（採用） — 実数入力専用、出力が N/2+1 点で効率的
  3. `scipy.fft.rfft` — numpy 互換でわずかに高速
- **Selected Approach**: `numpy.fft.rfft` を使用し、周波数軸は `numpy.fft.rfftfreq(n, d=0.05)` で計算する（5cm ピッチ → サンプル間隔 0.05m）。
- **Rationale**: 実数データに特化した rfft がメモリと計算効率で優れる。numpy は既存依存関係として必ず存在する。
- **Trade-offs**: scipy.fft との差はわずかで、依存を増やさない numpy を優先。

---

## Risks & Mitigations

- **日本語テキスト列のエンコーディング問題** — `pandas.read_excel` は openpyxl バックエンドで UTF-8/Shift-JIS を自動処理するが、文字化け発生時のフォールバックを DataLoader のエラー処理に組み込む
- **大規模データでのメモリ使用量** — 5cm ピッチ 1 区間分で数万〜数十万行になる可能性がある。`ChunkReader` パターンを将来拡張として設計に明示する
- **ウィンドウ幅の不正値入力** — 要件 3.6・5.6 に対応する入力バリデーションを `ParameterValidator` に一元化してバグ混入を防ぐ
- **Savitzky-Golay フィルタの偶数ウィンドウ** — `scipy.signal.savgol_filter` はウィンドウ幅が奇数でなければならない。偶数入力時に自動で +1 補正するロジックを組み込む

---

## References

- [Plotly Python 公式ドキュメント](https://plotly.com/python/) — サブプロット、HTML 出力
- [SciPy Signal 処理リファレンス](https://docs.scipy.org/doc/scipy/reference/signal.html) — savgol_filter, stft
- [NumPy FFT リファレンス](https://numpy.org/doc/stable/reference/routines.fft.html) — rfft, rfftfreq
- [Pandas Rolling API](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rolling.html) — rolling median, rolling mean
