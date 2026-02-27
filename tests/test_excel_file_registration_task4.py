"""タスク 4: データソース切り替えとパイプライン統合のテスト

タスク 4.1: 選択されたデータソースに応じたデータ取得ロジックを実装する
タスク 4.2: 現在使用中のデータソースをメインエリアに表示する
Requirements: 1.6, 4.1, 4.4, 5.1, 5.2, 5.3
"""
from __future__ import annotations

from pathlib import Path

import pytest


APP_PY_PATH = Path("/home/sagemaker-user/5cm-chart-analysis/app.py")


def read_app_source() -> str:
    """app.py のソースコードを文字列として返す。"""
    return APP_PY_PATH.read_text(encoding="utf-8")


# ─────────────────────────────────────────────
# タスク 4.1: データソース選択に応じたデータ取得ロジック
# ─────────────────────────────────────────────


class TestDataSourceRoutingLogic:
    """4.1 データソース選択に応じたデータ取得ロジックを検証する（要件 1.6, 4.1, 4.4, 5.1, 5.2）。"""

    def test_upload_branch_calls_load_uploaded_data(self) -> None:
        """「アップロードファイルを使用」ブロック内で load_uploaded_data が呼び出される（要件 4.1）。"""
        source = read_app_source()
        lines = source.split("\n")

        # data_source の条件分岐が複数ある場合（サイドバー + データ読み込み）、
        # すべての出現を調べ、少なくとも1つのブロック内に load_uploaded_data があることを確認する
        upload_block_starts = [
            i for i, line in enumerate(lines)
            if (
                "アップロードファイルを使用" in line
                and "if" in line
                and "data_source" in line
            )
        ]

        assert len(upload_block_starts) > 0, (
            "app.py に data_source == 'アップロードファイルを使用' の条件分岐が必要です"
        )

        # いずれかのブロック内に load_uploaded_data の呼び出しがあること
        found = False
        for upload_block_start in upload_block_starts:
            upload_block_lines = []
            for line in lines[upload_block_start:upload_block_start + 50]:
                stripped = line.strip()
                if stripped.startswith("else") and stripped.endswith(":"):
                    break
                upload_block_lines.append(line)
            if "load_uploaded_data(" in "\n".join(upload_block_lines):
                found = True
                break

        assert found, (
            "「アップロードファイルを使用」ブロック内で load_uploaded_data() を呼び出す必要があります（要件 4.1）"
        )

    def test_default_branch_calls_load_data(self) -> None:
        """「デフォルトファイルを使用」ブロック内で load_data が呼び出される（要件 1.6, 5.1）。"""
        source = read_app_source()
        lines = source.split("\n")

        upload_if_lines = [
            i for i, line in enumerate(lines)
            if "アップロードファイルを使用" in line and "if" in line and "data_source" in line
        ]
        assert len(upload_if_lines) > 0, "data_source の条件分岐が見つかりません"

        # data_source のいずれかの条件分岐から 100 行以内に load_data( の呼び出しがあること
        found = False
        for if_start in upload_if_lines:
            context = "\n".join(lines[if_start:if_start + 100])
            if "load_data(" in context:
                found = True
                break

        assert found, (
            "デフォルトファイルブロック内で load_data() を呼び出す必要があります（要件 1.6, 5.1）"
        )

    def test_df_variable_used_in_pipeline(self) -> None:
        """データ読み込み後に df 変数が割り当てられ、パイプラインで使用される（要件 4.1）。"""
        source = read_app_source()
        assert "NoiseFilter" in source, (
            "データ読み込み後の df が NoiseFilter などのパイプラインに渡される必要があります（要件 4.1）"
        )

    def test_existing_parameter_widgets_unchanged(self) -> None:
        """既存の分析パラメータウィジェット定義が変更されていない（要件 4.4）。"""
        source = read_app_source()
        assert "ウィンドウ幅" in source, "ウィンドウ幅ウィジェットが必要です（要件 4.4）"
        assert "Z-Score 閾値" in source, "Z-Score 閾値ウィジェットが必要です（要件 4.4）"
        assert "移動中央値フィルタ" in source, "移動中央値フィルタウィジェットが必要です（要件 4.4）"
        assert "Savitzky-Golay フィルタ" in source, "Savitzky-Golay フィルタウィジェットが必要です（要件 4.4）"

    def test_default_data_path_constant_defined(self) -> None:
        """DEFAULT_DATA_PATH 定数がモジュールレベルで定義されている（要件 1.6, 5.1）。"""
        source = read_app_source()
        assert "DEFAULT_DATA_PATH" in source, (
            "app.py に DEFAULT_DATA_PATH 定数が必要です（デフォルトデータファイルパス）"
        )

    def test_upload_file_used_in_upload_branch(self) -> None:
        """「アップロードファイルを使用」選択時に uploaded_file が使用される（要件 5.2）。"""
        source = read_app_source()
        lines = source.split("\n")

        upload_block_starts = [
            i for i, line in enumerate(lines)
            if (
                "アップロードファイルを使用" in line
                and "if" in line
                and "data_source" in line
            )
        ]
        assert len(upload_block_starts) > 0, "data_source の条件分岐が見つかりません"

        # いずれかのブロック内に uploaded_file の参照があること
        found = False
        for upload_block_start in upload_block_starts:
            upload_block_lines = []
            for line in lines[upload_block_start:upload_block_start + 50]:
                stripped = line.strip()
                if stripped.startswith("else") and stripped.endswith(":"):
                    break
                upload_block_lines.append(line)
            if "uploaded_file" in "\n".join(upload_block_lines):
                found = True
                break

        assert found, (
            "「アップロードファイルを使用」ブロック内で uploaded_file を使用する必要があります（要件 5.2）"
        )

    def test_pipeline_integration_full_chain(self) -> None:
        """NoiseFilter → AnomalyDetector → SignalAnalyzer → Visualizer のパイプライン呼び出しが存在する（要件 4.1）。"""
        source = read_app_source()
        assert "NoiseFilter" in source, "NoiseFilter がパイプラインに含まれている必要があります"
        assert "AnomalyDetector" in source, "AnomalyDetector がパイプラインに含まれている必要があります"
        assert "SignalAnalyzer" in source, "SignalAnalyzer がパイプラインに含まれている必要があります"
        assert "Visualizer" in source, "Visualizer がパイプラインに含まれている必要があります"


# ─────────────────────────────────────────────
# タスク 4.2: 現在使用中データソースの表示
# ─────────────────────────────────────────────


class TestDataSourceIndicator:
    """4.2 現在使用中のデータソースをメインエリアに表示することを検証する（要件 5.3）。"""

    def test_default_data_label_exists_in_source(self) -> None:
        """「デフォルトデータ」ラベルが app.py に存在する（要件 5.3）。"""
        source = read_app_source()
        assert "デフォルトデータ" in source, (
            "app.py に「デフォルトデータ」というラベルのデータソース表示コードが必要です（要件 5.3）"
        )

    def test_uploaded_filename_used_in_indicator(self) -> None:
        """アップロードファイル使用時にファイル名（uploaded_file.name）が表示に使用される（要件 5.3）。"""
        source = read_app_source()
        # データソース表示コードで uploaded_file.name が参照されていること
        assert "uploaded_file.name" in source, (
            "app.py でアップロードファイル名（uploaded_file.name）をデータソース表示に使用する必要があります（要件 5.3）"
        )

    def test_data_source_indicator_in_main_area(self) -> None:
        """データソース表示がメインエリア（サイドバー外）で行われる（要件 5.3）。"""
        source = read_app_source()
        lines = source.split("\n")

        # 「デフォルトデータ」を含む行がサイドバー以外で使用されていること
        found_main_area = False
        for line in lines:
            if "デフォルトデータ" in line and "st.sidebar" not in line:
                found_main_area = True
                break

        assert found_main_area, (
            "データソース表示（「デフォルトデータ」）がメインエリア（st.sidebar なし）で行われる必要があります（要件 5.3）"
        )

    def test_data_source_indicator_shows_both_cases(self) -> None:
        """アップロードとデフォルトの両方のケースでデータソース表示が存在する（要件 5.3）。"""
        source = read_app_source()
        has_upload_indicator = "uploaded_file.name" in source
        has_default_indicator = "デフォルトデータ" in source
        assert has_upload_indicator and has_default_indicator, (
            "アップロードファイル使用時（uploaded_file.name）と"
            "デフォルト使用時（「デフォルトデータ」）の両方のデータソース表示が必要です（要件 5.3）"
        )

    def test_data_source_indicator_uses_display_function(self) -> None:
        """データソース表示が Streamlit の表示関数（st.caption / st.info / st.write）を使用する（要件 5.3）。"""
        source = read_app_source()
        lines = source.split("\n")

        # 「デフォルトデータ」テキストの周辺（前後 5 行）に Streamlit 表示関数があること
        found_display = False
        for i, line in enumerate(lines):
            if "デフォルトデータ" in line:
                context = "\n".join(lines[max(0, i - 3):i + 3])
                if any(fn in context for fn in ["st.caption(", "st.info(", "st.write(", "st.markdown("]):
                    found_display = True
                    break

        assert found_display, (
            "データソース表示に st.caption() / st.info() / st.write() / st.markdown() が必要です（要件 5.3）"
        )
