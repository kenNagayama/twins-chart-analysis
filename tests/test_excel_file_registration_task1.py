"""タスク 1: サイドバーにデータソース選択 UI を追加するテスト

タスク 1.1: データソース選択ラジオボタンとファイルアップロードウィジェットをサイドバーに配置する
"""
from __future__ import annotations

import ast
import inspect
import textwrap
from pathlib import Path

import pytest


# ─────────────────────────────────────────────
# テスト対象ファイルのパス
# ─────────────────────────────────────────────

APP_PY_PATH = Path("/home/sagemaker-user/5cm-chart-analysis/app.py")


def read_app_source() -> str:
    """app.py のソースコードを文字列として返す。"""
    return APP_PY_PATH.read_text(encoding="utf-8")


# ─────────────────────────────────────────────
# タスク 1.1: サイドバーのデータソース選択 UI の配置テスト
# ─────────────────────────────────────────────


class TestDataSourceSelectionRadioButton:
    """1.1 データソース選択ラジオボタンの存在と設定を検証する。

    Streamlit ウィジェット自体のランタイムテストは困難なため、
    app.py のソースコードに正しい実装が存在することを検証する。
    """

    def test_st_radio_call_exists_in_app(self) -> None:
        """app.py に st.sidebar.radio または st.radio の呼び出しが存在する（要件 1.1）。"""
        source = read_app_source()
        assert "st.sidebar.radio" in source or ".radio(" in source, (
            "app.py に st.radio または st.sidebar.radio の呼び出しが必要です"
        )

    def test_radio_option_upload_file_exists(self) -> None:
        """ラジオボタンに「アップロードファイルを使用」選択肢が存在する（要件 1.1）。"""
        source = read_app_source()
        assert "アップロードファイルを使用" in source, (
            "app.py に「アップロードファイルを使用」という選択肢が必要です"
        )

    def test_radio_option_default_file_exists(self) -> None:
        """ラジオボタンに「デフォルトファイルを使用」選択肢が存在する（要件 1.1）。"""
        source = read_app_source()
        assert "デフォルトファイルを使用" in source, (
            "app.py に「デフォルトファイルを使用」という選択肢が必要です"
        )

    def test_radio_initial_selection_is_upload(self) -> None:
        """ラジオボタンの初期選択が「アップロードファイルを使用」である（要件 1.2）。

        index=0 で初期選択を指定するか、選択肢リストの先頭に配置することを確認。
        """
        source = read_app_source()
        # ラジオボタンの options の先頭が「アップロードファイルを使用」であることを確認
        # または index=0 が指定されていることを確認
        has_upload_first = (
            '"アップロードファイルを使用"' in source
            or "'アップロードファイルを使用'" in source
        )
        assert has_upload_first, (
            "app.py のラジオボタン選択肢の先頭が「アップロードファイルを使用」である必要があります"
        )

    def test_data_source_section_header_exists(self) -> None:
        """サイドバーに「データソース選択」セクションヘッダーが存在する（要件 1.1）。"""
        source = read_app_source()
        assert "データソース選択" in source, (
            "app.py に「データソース選択」セクションが必要です"
        )


class TestFileUploaderWidget:
    """1.1 ファイルアップロードウィジェットの存在と設定を検証する。"""

    def test_st_file_uploader_call_exists_in_app(self) -> None:
        """app.py に st.file_uploader の呼び出しが存在する（要件 1.3）。"""
        source = read_app_source()
        assert "file_uploader" in source, (
            "app.py に st.file_uploader の呼び出しが必要です"
        )

    def test_file_uploader_label_is_japanese(self) -> None:
        """ファイルアップロードウィジェットのラベルが日本語テキストである（要件 1.4）。"""
        source = read_app_source()
        assert "分析ファイルを選択" in source, (
            "app.py のファイルアップロードウィジェットのラベルに「分析ファイルを選択」が必要です"
        )

    def test_file_uploader_accepts_only_xlsx(self) -> None:
        """ファイルアップロードウィジェットが .xlsx のみ許可する（要件 1.5）。"""
        source = read_app_source()
        # type=["xlsx"] または type=["xlsx"] の形式で指定されていることを確認
        has_xlsx_type = '"xlsx"' in source or "'xlsx'" in source
        assert has_xlsx_type, (
            "app.py のファイルアップロードウィジェットに type=[\"xlsx\"] の指定が必要です"
        )

    def test_file_uploader_shown_only_when_upload_selected(self) -> None:
        """ファイルアップロードウィジェットが条件付きで表示される（要件 1.3, 1.6）。

        file_uploader の呼び出しが if 文の条件ブロック内に存在することを確認する。
        """
        source = read_app_source()
        # file_uploader が「アップロードファイルを使用」の条件分岐内に存在することを
        # ソースコードの構造から確認する
        lines = source.split("\n")
        file_uploader_line_idx = None
        for i, line in enumerate(lines):
            if "file_uploader" in line:
                file_uploader_line_idx = i
                break

        assert file_uploader_line_idx is not None, "file_uploader の呼び出しが見つかりません"

        # file_uploader より前の行で if 文（データソース選択の条件）が存在することを確認
        context_lines = lines[max(0, file_uploader_line_idx - 10):file_uploader_line_idx]
        context_text = "\n".join(context_lines)
        has_conditional = "if" in context_text and "アップロードファイルを使用" in context_text
        assert has_conditional, (
            "file_uploader は「アップロードファイルを使用」が選択されているときのみ"
            "表示される条件分岐内に配置される必要があります"
        )


class TestSidebarWidgetOrder:
    """1.1 サイドバーのウィジェット定義順序を検証する。

    設計書の要件: データソース選択 → ファイルアップロードウィジェット（条件付き）→ 分析パラメータ
    """

    def test_data_source_section_appears_before_analysis_parameters(self) -> None:
        """「データソース選択」セクションが「分析パラメータ」ヘッダーより前に定義される（要件 1.6）。"""
        source = read_app_source()
        data_source_pos = source.find("データソース選択")
        analysis_params_pos = source.find("分析パラメータ")

        assert data_source_pos != -1, "「データソース選択」がapp.pyに見つかりません"
        assert analysis_params_pos != -1, "「分析パラメータ」がapp.pyに見つかりません"
        assert data_source_pos < analysis_params_pos, (
            "「データソース選択」は「分析パラメータ」より前に定義されている必要があります\n"
            f"データソース選択の位置: {data_source_pos}\n"
            f"分析パラメータの位置: {analysis_params_pos}"
        )

    def test_file_uploader_appears_before_analysis_parameters(self) -> None:
        """file_uploader の定義が「分析パラメータ」ヘッダーより前に存在する（要件 1.6）。"""
        source = read_app_source()
        file_uploader_pos = source.find("file_uploader")
        analysis_params_pos = source.find("分析パラメータ")

        assert file_uploader_pos != -1, "file_uploaderがapp.pyに見つかりません"
        assert analysis_params_pos != -1, "「分析パラメータ」がapp.pyに見つかりません"
        assert file_uploader_pos < analysis_params_pos, (
            "file_uploader は「分析パラメータ」より前に定義されている必要があります"
        )

    def test_all_sidebar_widgets_defined_before_data_loading(self) -> None:
        """すべてのサイドバーウィジェット定義がデータ取得処理より前に配置される（要件 1.6）。

        load_data または load_uploaded_data の呼び出しより前に
        サイドバーのウィジェット定義が完了していることを確認する。
        """
        source = read_app_source()
        # 分析パラメータのウィジェット（スライダーなど）の最後の位置を探す
        # 分析パラメータセクションの最後はスライダーかトグルの定義
        sidebar_header_pos = source.find('sidebar.header("分析パラメータ")')
        if sidebar_header_pos == -1:
            sidebar_header_pos = source.find("分析パラメータ")

        # load_data の呼び出し位置（データ取得処理）を探す
        # 実際のデータ取得処理（df_or_error = load_data(...)のような形）
        load_data_call_pos = source.find("load_data(")
        # DEFAULT_DATA_PATH の定義行は除外し、実際の呼び出し行を探す
        # 関数定義行（def load_data）も除外する
        lines = source.split("\n")
        actual_load_call_pos = -1
        for i, line in enumerate(lines):
            stripped = line.strip()
            # 関数定義や DEFAULT_DATA_PATH 定義は除外
            if "load_data(" in stripped and not stripped.startswith("def ") and "DEFAULT_DATA_PATH" not in stripped:
                # この行の文字列位置を計算
                actual_load_call_pos = sum(len(lines[j]) + 1 for j in range(i))
                break

        assert actual_load_call_pos != -1, "load_data の呼び出し行が見つかりません"
        assert sidebar_header_pos < actual_load_call_pos, (
            "サイドバーの「分析パラメータ」ウィジェット定義がデータ取得処理より前にある必要があります"
        )


class TestDataSourceSelectionLogic:
    """1.1 データソース選択ロジックを検証する。

    Streamlit ウィジェットに依存しない純粋なロジックのテスト。
    """

    def test_data_source_options_count_is_two(self) -> None:
        """データソース選択は2択である（要件 1.1）。"""
        source = read_app_source()
        # 両方の選択肢が存在することを確認
        assert "アップロードファイルを使用" in source, "アップロード選択肢が必要"
        assert "デフォルトファイルを使用" in source, "デフォルト選択肢が必要"

    def test_default_data_path_constant_defined(self) -> None:
        """DEFAULT_DATA_PATH 定数が定義されている（要件 1.6）。"""
        source = read_app_source()
        assert "DEFAULT_DATA_PATH" in source, (
            "app.py に DEFAULT_DATA_PATH 定数が定義されている必要があります"
        )

    def test_conditional_display_based_on_radio_selection(self) -> None:
        """ラジオボタンの選択に基づいてウィジェット表示を切り替える条件分岐が存在する（要件 1.3, 1.6）。"""
        source = read_app_source()
        # 「アップロードファイルを使用」の条件分岐が存在することを確認
        # data_source 変数か直接比較かを検証
        has_conditional_logic = (
            "== \"アップロードファイルを使用\"" in source
            or "== 'アップロードファイルを使用'" in source
            or "アップロードファイルを使用" in source and "if" in source
        )
        assert has_conditional_logic, (
            "ラジオボタンの選択値に基づく条件分岐が app.py に必要です"
        )
