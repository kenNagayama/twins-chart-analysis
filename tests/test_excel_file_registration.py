"""タスク 2: アップロードファイル読み込み用のキャッシュ関数テスト

load_uploaded_data キャッシュ関数の構造検証と動作テスト。
Requirements: 4.2, 4.3
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import pytest

from src.data_loader import REQUIRED_COLUMNS, DataLoader, LoadError


APP_PY_PATH = Path("/home/sagemaker-user/5cm-chart-analysis/app.py")


def read_app_source() -> str:
    """app.py のソースコードを文字列として返す。"""
    return APP_PY_PATH.read_text(encoding="utf-8")


# ─────────────────────────────────────────────
# テスト用ヘルパー
# ─────────────────────────────────────────────


def make_valid_xlsx_bytes() -> bytes:
    """有効な xlsx ファイルのバイト列を生成する。"""
    data = {col: ["dummy_value"] for col in REQUIRED_COLUMNS}
    data["CH"] = [1]
    buf = io.BytesIO()
    pd.DataFrame(data).to_excel(buf, index=False, engine="openpyxl")
    buf.seek(0)
    return buf.read()


def make_missing_columns_xlsx_bytes() -> bytes:
    """必須列が欠損した xlsx ファイルのバイト列を生成する。"""
    buf = io.BytesIO()
    pd.DataFrame({"dummy_col": [1, 2, 3]}).to_excel(buf, index=False, engine="openpyxl")
    buf.seek(0)
    return buf.read()


class MockUploadedFile(io.BytesIO):
    """st.file_uploader の戻り値（UploadedFile）のモック。

    io.BytesIO を継承して pandas / openpyxl が要求するファイルインターフェースを完全に提供する。
    """

    def __init__(self, name: str, content: bytes) -> None:
        super().__init__(content)
        self.name = name
        self.file_id = f"mock-file-id-{name}"


# ─────────────────────────────────────────────
# load_uploaded_data 関数の構造テスト
# ─────────────────────────────────────────────


class TestLoadUploadedDataStructure:
    """load_uploaded_data 関数の定義構造を検証する（要件 4.2, 4.3）。"""

    def test_function_defined_in_app(self) -> None:
        """app.py に load_uploaded_data 関数が定義されている（要件 4.2）。"""
        source = read_app_source()
        assert "def load_uploaded_data(" in source, (
            "app.py に load_uploaded_data 関数が必要です"
        )

    def test_cache_data_decorator_applied(self) -> None:
        """load_uploaded_data に @st.cache_data デコレータが適用されている（要件 4.2）。"""
        source = read_app_source()
        func_idx = source.find("def load_uploaded_data(")
        assert func_idx != -1, "load_uploaded_data 関数が見つかりません"

        # 関数定義の直前に @st.cache_data があることを確認
        context = source[max(0, func_idx - 120):func_idx]
        assert "cache_data" in context, (
            "load_uploaded_data 関数の直前に @st.cache_data デコレータが必要です"
        )

    def test_hash_funcs_specified_with_file_id(self) -> None:
        """@st.cache_data に hash_funcs で file_id が指定されている（要件 4.3）。"""
        source = read_app_source()
        func_idx = source.find("def load_uploaded_data(")
        assert func_idx != -1, "load_uploaded_data 関数が見つかりません"

        # デコレータ部分（関数定義の前）に hash_funcs と file_id があることを確認
        context = source[max(0, func_idx - 200):func_idx]
        assert "hash_funcs" in context, (
            "@st.cache_data に hash_funcs パラメータが必要です"
        )
        assert "file_id" in context, (
            "hash_funcs に UploadedFile の file_id を使用するキャッシュキーの指定が必要です"
        )

    def test_calls_load_from_upload_in_body(self) -> None:
        """load_uploaded_data の内部で load_from_upload() を呼び出す（要件 4.2）。"""
        source = read_app_source()
        func_start = source.find("def load_uploaded_data(")
        assert func_start != -1, "load_uploaded_data 関数が見つかりません"

        # 関数定義から次の関数定義または空行区切りまでを確認
        func_body = source[func_start:func_start + 400]
        assert "load_from_upload" in func_body, (
            "load_uploaded_data 内で DataLoader().load_from_upload() の呼び出しが必要です"
        )

    def test_function_at_module_level(self) -> None:
        """load_uploaded_data がモジュールレベルに定義されている（要件 4.2）。"""
        source = read_app_source()
        lines = source.split("\n")
        for i, line in enumerate(lines):
            if "def load_uploaded_data(" in line:
                assert not line.startswith(" ") and not line.startswith("\t"), (
                    f"load_uploaded_data はモジュールレベル（インデントなし）に定義されるべきです。"
                    f"行 {i + 1}: {line!r}"
                )
                break
        else:
            pytest.fail("load_uploaded_data 関数が app.py に見つかりません")


# ─────────────────────────────────────────────
# load_uploaded_data の動作テスト（DataLoader 経由）
# ─────────────────────────────────────────────


class TestLoadUploadedDataBehavior:
    """load_uploaded_data の動作を DataLoader.load_from_upload 経由で検証する（要件 4.2, 4.3）。

    load_uploaded_data は DataLoader.load_from_upload の薄いラッパーのため、
    @st.cache_data をバイパスして DataLoader を直接呼び出すことで動作を検証する。
    """

    def _invoke(self, file_obj: MockUploadedFile) -> pd.DataFrame | LoadError:
        """DataLoader.load_from_upload を直接呼び出す（load_uploaded_data の内部ロジック）。"""
        return DataLoader().load_from_upload(file_obj, file_obj.name)

    def test_valid_xlsx_returns_dataframe(self) -> None:
        """有効な xlsx ファイルオブジェクトで DataFrame が返る（要件 2.1, 4.2）。"""
        mock_file = MockUploadedFile("test.xlsx", make_valid_xlsx_bytes())
        result = self._invoke(mock_file)
        assert isinstance(result, pd.DataFrame), (
            f"有効な xlsx ファイルで DataFrame が返る必要があります。実際の戻り値: {type(result)}"
        )

    def test_invalid_format_returns_load_error(self) -> None:
        """非 xlsx ファイルオブジェクトで LoadError(kind='invalid_format') が返る（要件 2.3）。"""
        mock_file = MockUploadedFile("test.csv", b"col1,col2\n1,2")
        result = self._invoke(mock_file)
        assert isinstance(result, LoadError), "非 xlsx ファイルは LoadError を返す必要があります"
        assert result.kind == "invalid_format", (
            f"非 xlsx は LoadError(kind='invalid_format') を返すべきです。実際: {result.kind}"
        )

    def test_missing_columns_returns_load_error(self) -> None:
        """必須列が欠損したファイルで LoadError(kind='missing_columns') が返る（要件 2.4）。"""
        mock_file = MockUploadedFile("missing_cols.xlsx", make_missing_columns_xlsx_bytes())
        result = self._invoke(mock_file)
        assert isinstance(result, LoadError), "必須列欠損ファイルは LoadError を返す必要があります"
        assert result.kind == "missing_columns", (
            f"必須列欠損ファイルは LoadError(kind='missing_columns') を返すべきです。実際: {result.kind}"
        )

    def test_does_not_raise_exception(self) -> None:
        """例外を外部に送出しない（要件 2.6）。"""
        mock_file = MockUploadedFile("broken.xlsx", b"not valid xlsx content at all")
        try:
            result = self._invoke(mock_file)
            assert isinstance(result, LoadError), "例外ではなく LoadError を返す必要があります"
        except Exception as e:
            pytest.fail(f"例外が発生しました（例外を送出しないべきです）: {e}")
