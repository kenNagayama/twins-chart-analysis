"""DataLayer: Excel読み込み・列検証・CH分割"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# 必須列リスト（要件 1.2）
REQUIRED_COLUMNS: list[str] = [
    "キロ程",
    "摩耗_測定値",
    "CH",
    "箇所名",
    "通称線名名称",
    "駅・駅々間名称",
    "電柱番号",
    "架線構造名",
    "トロリ線種",
    "降雨フラグ",
]

# CH 列の有効範囲
_CH_VALID_RANGE = range(1, 5)  # 1, 2, 3, 4


@dataclass(frozen=True)
class LoadError:
    """データ読み込み・列検証のエラー値オブジェクト。

    Attributes:
        kind: エラー種別。"file_not_found" | "missing_columns" | "read_error"
        message: ユーザ向けエラーメッセージ
        missing_columns: 欠損列名リスト（kind="missing_columns" のときに使用）
    """

    kind: str
    message: str
    missing_columns: list[str] = field(default_factory=list)


class DataLoader:
    """Excel ファイルの読み込み・列検証・CH 別 DataFrame 提供を担当するクラス。

    エラー時は例外を外部に送出せず、LoadError を値として返す。
    """

    def load(self, file_path: str) -> pd.DataFrame | LoadError:
        """Excel ファイルを読み込み DataFrame を返す。

        Args:
            file_path: 読み込む Excel ファイルのパス（非空文字列）

        Returns:
            成功時は必須列をすべて含む pd.DataFrame。
            エラー時は LoadError を返す（例外を送出しない）。
        """
        path = Path(file_path)

        # ファイル存在確認（要件 1.3）
        if not path.exists():
            return LoadError(
                kind="file_not_found",
                message=f"ファイルが見つかりません: {path}",
            )

        # Excel 読み込み（要件 1.1）
        try:
            df = pd.read_excel(path, engine="openpyxl")
        except Exception as exc:
            return LoadError(
                kind="read_error",
                message=f"ファイルの読み込みに失敗しました: {exc}",
            )

        # 列名の正規化: 全角「ＣＨ」→ 半角「CH」
        df = df.rename(columns={"ＣＨ": "CH"})

        # 列検証（要件 1.2）
        validation_error = self.validate_columns(df)
        if validation_error is not None:
            return validation_error

        return df

    def validate_columns(self, df: pd.DataFrame) -> Optional[LoadError]:
        """必須列の存在を検証する。

        Args:
            df: 検証対象の DataFrame

        Returns:
            全必須列が存在する場合は None。
            欠損列がある場合は LoadError(kind="missing_columns") を返す。
        """
        missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
        if not missing:
            return None

        cols_str = ", ".join(missing)
        return LoadError(
            kind="missing_columns",
            message=f"必須列が不足しています: {cols_str}",
            missing_columns=missing,
        )

    # ------------------------------------------------------------------
    # 将来拡張: ファイルアップロード機能の拡張点（要件 7.1〜7.4）
    # ------------------------------------------------------------------
    # このメソッドを使うことで、既存の load(file_path) や下流のパイプライン
    # コード（NoiseFilter, AnomalyDetector, Visualizer 等）を一切変更せずに
    # Streamlit の st.file_uploader 経由のデータ読み込みを追加できる。
    #
    # 将来の統合イメージ（app.py に追加するだけでよい）:
    #   uploaded = st.file_uploader("ファイルを選択", type=["xlsx"])
    #   if uploaded is not None:
    #       result = loader.load_from_upload(uploaded, uploaded.name)
    #       # 以降は load() の戻り値と同じ型なので既存パイプラインをそのまま使用可能
    # ------------------------------------------------------------------

    def load_from_upload(self, file_obj: IO[bytes], filename: str) -> pd.DataFrame | LoadError:
        """将来拡張: ファイルオブジェクトから Excel データを読み込む（要件 7.1〜7.4）。

        ``st.file_uploader`` が返すファイルオブジェクトを受け付けられる拡張インターフェース。
        戻り値の型は :meth:`load` と同一のため、既存パイプラインコードを変更せずに
        アップロード機能を追加できる。

        Args:
            file_obj: 読み込むファイルオブジェクト（バイナリ）。
                      ``st.file_uploader`` の戻り値や ``io.BytesIO`` を渡せる。
            filename: ファイル名（拡張子検証に使用）。
                      ``st.file_uploader`` では ``uploaded_file.name`` を渡す。

        Returns:
            成功時は必須列をすべて含む ``pd.DataFrame``。
            エラー時は ``LoadError`` を返す（例外を送出しない）。

            - ``kind="invalid_format"``: .xlsx 以外のファイル（要件 7.4）
            - ``kind="read_error"``: Excel 解析エラー
            - ``kind="missing_columns"``: 必須列が欠損（要件 7.2）
        """
        # 拡張子チェック: .xlsx 以外は非対応形式エラー（要件 7.4）
        if not filename.lower().endswith(".xlsx"):
            return LoadError(
                kind="invalid_format",
                message=(
                    f"非対応のファイル形式です: {filename}。"
                    ".xlsx ファイルを指定してください。"
                ),
            )

        # Excel 読み込み（ファイルオブジェクトから直接）
        try:
            df = pd.read_excel(file_obj, engine="openpyxl")
        except Exception as exc:
            return LoadError(
                kind="read_error",
                message=f"ファイルの読み込みに失敗しました: {exc}",
            )

        # 列検証: load() と同一ロジックを再利用（要件 7.2）
        validation_error = self.validate_columns(df)
        if validation_error is not None:
            return validation_error

        return df

    def get_channel_group(self, df: pd.DataFrame, ch: int) -> pd.DataFrame:
        """CH 列の値でフィルタリングした DataFrame を返す。

        Args:
            df: フィルタリング対象の DataFrame（必須列含む）
            ch: 抽出する CH 値（有効範囲: 1〜4）

        Returns:
            CH == ch の行のみ含む DataFrame。
            ch が 1〜4 の範囲外の場合は空の DataFrame を返してログに記録する。
        """
        if ch not in _CH_VALID_RANGE:
            logger.warning(
                "CH 値 %d は有効範囲 1〜4 の外です。空の DataFrame を返します。",
                ch,
            )
            return df.iloc[0:0].copy()

        return df[df["CH"] == ch]
