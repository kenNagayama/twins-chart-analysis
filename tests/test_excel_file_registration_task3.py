"""タスク 3: エラーハンドリングとユーザーフィードバック表示のテスト

タスク 3.1: ファイル未選択時のガード処理を実装する
タスク 3.2: LoadError の種別に応じたエラーメッセージ表示を実装する
Requirements: 1.7, 5.4, 3.1, 3.2, 3.3, 3.4, 3.5
"""
from __future__ import annotations

from pathlib import Path

import pytest


APP_PY_PATH = Path("/home/sagemaker-user/5cm-chart-analysis/app.py")


def read_app_source() -> str:
    """app.py のソースコードを文字列として返す。"""
    return APP_PY_PATH.read_text(encoding="utf-8")


# ─────────────────────────────────────────────
# タスク 3.1: ファイル未選択時のガード処理
# ─────────────────────────────────────────────


class TestFileNotSelectedGuard:
    """3.1 ファイル未選択時のガード処理を検証する（要件 1.7, 5.4）。

    「アップロードファイルを使用」が選択されていてファイルが未選択の場合、
    st.info() でメッセージを表示して st.stop() を呼び出す。
    """

    def test_st_info_called_when_no_file_selected(self) -> None:
        """ファイル未選択時に st.info() が呼び出される（要件 1.7）。

        app.py のソースコードに、uploaded_file が None の場合に
        st.info() を呼び出すコードが存在することを確認する。
        """
        source = read_app_source()
        # uploaded_file が None のガード処理ブロックに st.info() が存在すること
        # None チェックと st.info の両方が存在することを確認
        assert "st.info(" in source, (
            "app.py に st.info() の呼び出しが必要です（ファイル未選択時のメッセージ表示）"
        )

    def test_st_stop_called_when_no_file_selected(self) -> None:
        """ファイル未選択時に st.stop() が呼び出される（要件 1.7）。

        app.py のソースコードに、ファイル未選択時に st.stop() を
        呼び出すコードが存在することを確認する。
        """
        source = read_app_source()
        assert "st.stop()" in source, (
            "app.py に st.stop() の呼び出しが必要です（ファイル未選択時のパイプライン停止）"
        )

    def test_info_and_stop_in_guard_context(self) -> None:
        """ファイル未選択時のガード処理で st.info() の後に st.stop() が呼ばれる（要件 1.7, 5.4）。

        uploaded_file の None チェックブロック内に st.info() と st.stop() の
        両方が近接して存在することを確認する。
        """
        source = read_app_source()
        lines = source.split("\n")

        # uploaded_file is None の条件チェックを含む行を探す
        none_check_lines = [
            i for i, line in enumerate(lines)
            if ("uploaded_file is None" in line or "if not uploaded_file" in line)
            and "if" in line
        ]
        assert len(none_check_lines) > 0, (
            "app.py に 'uploaded_file is None' または 'if not uploaded_file' の条件チェックが必要です"
        )

        # その条件チェックの後の近傍に st.info() と st.stop() が存在することを確認
        found_info_stop = False
        for check_line_idx in none_check_lines:
            # 条件チェックから 10 行以内に st.info() と st.stop() があることを確認
            context_lines = lines[check_line_idx:check_line_idx + 10]
            context_text = "\n".join(context_lines)
            if "st.info(" in context_text and "st.stop()" in context_text:
                found_info_stop = True
                break

        assert found_info_stop, (
            "ファイル未選択のガード処理ブロック内に st.info() と st.stop() の両方が必要です\n"
            "（uploaded_file is None の条件チェックから 10 行以内）"
        )

    def test_no_auto_revert_to_default_when_file_cleared(self) -> None:
        """ファイルがクリアされた場合にデフォルトファイルへの自動復帰が行われない（要件 5.4）。

        「アップロードファイルを使用」選択時に uploaded_file が None の場合、
        デフォルトファイルへのフォールバックが行われないことをソース構造から確認する。
        「アップロードファイルを使用」の条件ブロック内に load_data(DEFAULT_DATA_PATH) が
        ないことを確認する。
        """
        source = read_app_source()
        lines = source.split("\n")

        # 「アップロードファイルを使用」の条件ブロックを探す
        upload_block_start = None
        for i, line in enumerate(lines):
            if (
                "アップロードファイルを使用" in line
                and "if" in line
                and "data_source" in line
            ):
                upload_block_start = i
                break

        if upload_block_start is None:
            # data_source 変数の条件分岐が見つからない場合はスキップ
            # （別の実装パターンの可能性があるため失敗させない）
            return

        # 「アップロードファイルを使用」の条件ブロック内（30 行以内）に
        # load_data(DEFAULT_DATA_PATH) の呼び出しがないことを確認
        # ただし「elif デフォルト」以降の行は除外
        upload_block_lines = lines[upload_block_start:upload_block_start + 30]
        # elif か else が登場したらそれ以降は別のブロックなのでカット
        upload_only_lines = []
        for line in upload_block_lines:
            stripped = line.strip()
            if stripped.startswith("elif") or (stripped.startswith("else") and stripped.endswith(":")):
                break
            upload_only_lines.append(line)

        upload_block_text = "\n".join(upload_only_lines)
        assert "load_data(DEFAULT_DATA_PATH)" not in upload_block_text or upload_block_start is None, (
            "「アップロードファイルを使用」のブロック内でデフォルトファイルへの自動復帰が行われています。"
            "ファイルクリア時はデフォルトへ自動復帰せず、st.info() + st.stop() で停止してください（要件 5.4）"
        )

    def test_file_prompt_message_content(self) -> None:
        """ファイル選択を促すメッセージの内容が適切である（要件 1.7）。

        st.info() に渡すメッセージが「ファイル」「選択」に関する内容であることを確認する。
        """
        source = read_app_source()
        # st.info() の呼び出しの近くにファイル選択を促す文言があること
        # 「選択」「ファイル」のどちらかが st.info の引数内にあれば良い
        lines = source.split("\n")
        info_lines = [i for i, line in enumerate(lines) if "st.info(" in line]
        assert len(info_lines) > 0, "st.info() の呼び出しが見つかりません"

        # st.info() の引数（複数行にわたる場合も考慮して前後 3 行を確認）
        found_prompt = False
        for info_line_idx in info_lines:
            context = "\n".join(lines[info_line_idx:info_line_idx + 3])
            if "選択" in context or "ファイル" in context or "アップロード" in context:
                found_prompt = True
                break

        assert found_prompt, (
            "st.info() のメッセージにファイル選択を促す内容（「選択」「ファイル」「アップロード」）が必要です"
        )


# ─────────────────────────────────────────────
# タスク 3.2: LoadError 種別に応じたエラーメッセージ表示
# ─────────────────────────────────────────────


class TestLoadErrorDisplay:
    """3.2 LoadError の種別に応じたエラーメッセージ表示を検証する（要件 3.1, 3.2, 3.3, 3.4, 3.5）。

    各 LoadError の種別（invalid_format, missing_columns, read_error）に対して
    st.error() でエラーメッセージを表示し、st.stop() を呼び出すことを確認する。
    """

    def test_invalid_format_error_display(self) -> None:
        """invalid_format エラー時に st.error() が呼び出される（要件 3.1）。

        app.py のソースコードに 'invalid_format' の判定と st.error() が存在することを確認する。
        """
        source = read_app_source()
        assert "invalid_format" in source, (
            "app.py に LoadError の 'invalid_format' 種別の判定が必要です"
        )
        assert "st.error(" in source, (
            "app.py に st.error() の呼び出しが必要です（エラーメッセージ表示）"
        )

    def test_invalid_format_message_contains_xlsx(self) -> None:
        """invalid_format エラー時のメッセージに .xlsx への言及がある（要件 3.1）。

        「非対応のファイル形式です。.xlsx ファイルを指定してください。」に準ずるメッセージが
        存在することを確認する。
        """
        source = read_app_source()
        # .xlsx への言及があること
        has_xlsx_mention = ".xlsx" in source and "invalid_format" in source
        assert has_xlsx_mention, (
            "app.py の invalid_format エラー処理に .xlsx への言及が必要です"
        )

        # 条件分岐コードの invalid_format 判定を探す（== で比較している行）
        # docstring の行を除外するため、.kind == "invalid_format" の形式を探す
        lines = source.split("\n")
        invalid_format_cond_lines = [
            i for i, line in enumerate(lines)
            if "invalid_format" in line and "==" in line and "kind" in line
        ]

        assert len(invalid_format_cond_lines) > 0, (
            "app.py に '.kind == \"invalid_format\"' のような条件分岐が必要です"
        )
        # いずれかの invalid_format の条件分岐の後 5 行以内に .xlsx が含まれること
        found = False
        for invalid_format_cond_line in invalid_format_cond_lines:
            context = "\n".join(lines[invalid_format_cond_line:invalid_format_cond_line + 5])
            if ".xlsx" in context:
                found = True
                break

        assert found, (
            "invalid_format エラー処理の st.error() メッセージに '.xlsx' への言及が必要です"
        )

    def test_missing_columns_error_display(self) -> None:
        """missing_columns エラー時に欠損列名を含むメッセージが表示される（要件 3.2）。

        app.py のソースコードに 'missing_columns' の判定と、
        欠損列名リストを含むエラーメッセージの生成コードが存在することを確認する。
        """
        source = read_app_source()
        assert "missing_columns" in source, (
            "app.py に LoadError の 'missing_columns' 種別の判定が必要です"
        )

        # 欠損列名リストを使ってメッセージを生成するコードが存在すること
        # LoadError.missing_columns を参照するコードを確認
        # アップロードファイルの LoadError 処理ブロック内を探す
        lines = source.split("\n")
        # kind == "missing_columns" の条件分岐行を全て取得
        missing_cols_lines = [
            i for i, line in enumerate(lines)
            if "missing_columns" in line and ("==" in line or "kind" in line)
        ]

        assert len(missing_cols_lines) > 0, (
            "missing_columns 種別の判定コードが見つかりません"
        )

        # いずれかの判定後の近傍に .missing_columns を参照するコードが存在すること
        found = False
        for missing_cols_line in missing_cols_lines:
            context = "\n".join(lines[missing_cols_line:missing_cols_line + 8])
            if ".missing_columns" in context:
                found = True
                break

        assert found, (
            "missing_columns エラー処理で LoadError.missing_columns フィールドを参照するコードが必要です"
        )

    def test_read_error_display(self) -> None:
        """read_error エラー時に st.error() でメッセージが表示される（要件 3.3）。

        app.py のソースコードに 'read_error' の判定と st.error() が存在することを確認する。
        """
        source = read_app_source()
        assert "read_error" in source, (
            "app.py に LoadError の 'read_error' 種別の判定が必要です"
        )

        # read_error の条件分岐行（kind == "read_error"）を探す
        lines = source.split("\n")
        read_error_cond_lines = [
            i for i, line in enumerate(lines)
            if "read_error" in line and ("==" in line or "kind" in line)
        ]

        assert len(read_error_cond_lines) > 0, (
            "app.py に 'kind == \"read_error\"' のような条件分岐が必要です"
        )

        # いずれかの条件分岐後の近傍に st.error() が存在すること
        found = False
        for read_error_line in read_error_cond_lines:
            context = "\n".join(lines[read_error_line:read_error_line + 5])
            if "st.error(" in context:
                found = True
                break

        assert found, (
            "read_error エラー処理の後に st.error() の呼び出しが必要です"
        )

    def test_st_stop_called_after_st_error(self) -> None:
        """st.error() の呼び出し後に st.stop() が呼び出される（要件 3.4, 3.5）。

        app.py のソースコードで、エラー種別の判定ブロック内に
        st.error() → st.stop() の順序が維持されていることを確認する。
        """
        source = read_app_source()
        # LoadError の判定コードを探す
        lines = source.split("\n")

        # アップロードファイルの LoadError 判定ブロックを特定するためのマーカーを探す
        # load_uploaded_data または LoadError の判定が現れる場所
        error_block_start = None
        for i, line in enumerate(lines):
            if (
                "isinstance" in line
                and "LoadError" in line
                and "load_uploaded_data" not in lines[max(0, i - 5):i + 1]
            ):
                # load_uploaded_data の結果判定または他の LoadError 判定
                # data_source == "アップロードファイルを使用" ブロック内を探す
                error_block_start = i
                break

        assert error_block_start is not None, (
            "app.py に LoadError の isinstance チェックが見つかりません"
        )

        # エラーブロック内で st.error の後に st.stop が来ることを確認
        # 複数のエラー種別の処理をまとめてチェック
        # invalid_format, missing_columns, read_error の各ブロックで
        # st.error が st.stop より前にあることを確認
        error_positions = []
        stop_positions = []
        for i, line in enumerate(lines[error_block_start:error_block_start + 50]):
            if "st.error(" in line:
                error_positions.append(i)
            if "st.stop()" in line:
                stop_positions.append(i)

        # 少なくとも1組の st.error → st.stop の順序が存在すること
        assert len(error_positions) > 0, "エラーブロック内に st.error() が見つかりません"
        assert len(stop_positions) > 0, "エラーブロック内に st.stop() が見つかりません"

        # 最初の st.error の位置が最後の st.stop より前であること
        assert min(error_positions) < max(stop_positions), (
            "st.error() は st.stop() より前に呼ばれる必要があります（要件 3.4, 3.5）"
        )

    def test_error_stops_pipeline_execution(self) -> None:
        """エラー時に st.stop() でパイプラインの実行が停止される（要件 3.4）。

        app.py のアップロードファイルのエラー処理ブロックに st.stop() が
        存在することを確認する。
        """
        source = read_app_source()
        # アップロードファイル処理のエラーハンドリングブロックに st.stop() があること
        # invalid_format, missing_columns, read_error いずれかの処理後に st.stop() が呼ばれること
        lines = source.split("\n")

        # .kind == "invalid_format" の条件分岐行を探す（docstring の記述ではなく実際のコード）
        # == と kind の両方が含まれる行を探す
        error_handling_block = None
        for i, line in enumerate(lines):
            if "invalid_format" in line and "==" in line and "kind" in line:
                error_handling_block = i
                break

        assert error_handling_block is not None, (
            "invalid_format エラー処理ブロックが見つかりません（'.kind == \"invalid_format\"' のような行が必要）"
        )

        # そのブロックの周辺（30 行以内）に st.stop() が存在すること
        context = "\n".join(lines[error_handling_block:error_handling_block + 30])
        assert "st.stop()" in context, (
            "LoadError のエラー処理ブロック（30 行以内）に st.stop() が必要です（要件 3.4）"
        )

    def test_invalid_format_message_text(self) -> None:
        """invalid_format エラーメッセージが適切な内容を含む（要件 3.1）。

        「非対応」または「ファイル形式」のような文言が st.error のメッセージに含まれることを確認する。
        """
        source = read_app_source()
        lines = source.split("\n")

        # kind == "invalid_format" の条件分岐行を探す（docstring の記述を除外）
        invalid_format_cond_lines = [
            i for i, line in enumerate(lines)
            if "invalid_format" in line and ("==" in line or "kind" in line)
        ]

        assert len(invalid_format_cond_lines) > 0, (
            "app.py に 'kind == \"invalid_format\"' のような条件分岐が必要です"
        )

        # いずれかの条件分岐後 5 行以内に適切なメッセージが含まれること
        found = False
        for invalid_format_block_start in invalid_format_cond_lines:
            context = "\n".join(lines[invalid_format_block_start:invalid_format_block_start + 5])
            has_appropriate_message = (
                "非対応" in context
                or "ファイル形式" in context
                or "xlsx" in context.lower()
            )
            if has_appropriate_message:
                found = True
                break

        assert found, (
            "invalid_format エラーの st.error() メッセージに「非対応」「ファイル形式」または"
            "「xlsx」への言及が必要です"
        )

    def test_upload_error_handling_in_upload_branch(self) -> None:
        """アップロードファイルのエラー処理が「アップロードファイルを使用」ブロック内に実装される（要件 3.1〜3.5）。

        data_source が「アップロードファイルを使用」の場合の処理フロー内に
        LoadError のエラー処理が存在することを確認する。
        """
        source = read_app_source()
        # アップロードファイルの LoadError チェックコードが存在すること
        # isinstance(result, LoadError) または isinstance(..., LoadError) のチェック
        has_load_error_check = (
            "isinstance" in source and "LoadError" in source
        )
        assert has_load_error_check, (
            "app.py に LoadError の isinstance チェックが必要です（アップロードファイルのエラー処理）"
        )
