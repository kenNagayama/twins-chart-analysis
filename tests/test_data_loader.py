"""DataLoader のユニットテスト (タスク 2.1 / 2.2 / 2.3)"""
from __future__ import annotations

import logging
import os

import pandas as pd
import pytest

from src.data_loader import REQUIRED_COLUMNS, DataLoader, LoadError

# ---------------------------------------------------------------------------
# フィクスチャ
# ---------------------------------------------------------------------------

VALID_COLUMNS = list(REQUIRED_COLUMNS)  # 全10列


def make_valid_df(rows: int = 5) -> pd.DataFrame:
    """全必須列を含む小規模な DataFrame を生成するヘルパー。"""
    data = {
        "キロ程": [float(i) * 0.05 for i in range(rows)],
        "摩耗_測定値": [40.0 - i * 0.1 for i in range(rows)],
        "CH": [1, 2, 3, 4, 1][:rows],
        "箇所名": [f"箇所{i}" for i in range(rows)],
        "通称線名名称": [f"線名{i}" for i in range(rows)],
        "駅・駅々間名称": [f"駅間{i}" for i in range(rows)],
        "電柱番号": [f"電柱{i}" for i in range(rows)],
        "架線構造名": [f"架線{i}" for i in range(rows)],
        "トロリ線種": [f"線種{i}" for i in range(rows)],
        "降雨フラグ": ["なし"] * rows,
    }
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# 2.1: Excelファイルの読み込み処理
# ---------------------------------------------------------------------------


class TestLoadFileNotFound:
    """ファイルが存在しない場合は LoadError(kind="file_not_found") を返す。"""

    def test_missing_file_returns_load_error(self) -> None:
        loader = DataLoader()
        result = loader.load("/tmp/nonexistent_file_xyz.xlsx")

        assert isinstance(result, LoadError), "存在しないファイルは LoadError を返すべき"

    def test_missing_file_kind_is_file_not_found(self) -> None:
        loader = DataLoader()
        result = loader.load("/tmp/nonexistent_file_xyz.xlsx")

        assert isinstance(result, LoadError)
        assert result.kind == "file_not_found"

    def test_missing_file_does_not_raise_exception(self) -> None:
        """例外が外部に送出されないことを確認する。"""
        loader = DataLoader()
        # 例外が送出されればここでテスト失敗となる
        result = loader.load("/tmp/nonexistent_file_xyz.xlsx")
        assert result is not None


class TestLoadReadError:
    """読み込みエラー時は LoadError(kind="read_error") を返す。"""

    def test_invalid_file_returns_read_error(self, tmp_path: pytest.TempPathFactory) -> None:
        """Excel でないファイルを読もうとすると read_error を返す。"""
        # テキストファイルをあたかも xlsx として渡す
        fake_xlsx = tmp_path / "fake.xlsx"
        fake_xlsx.write_text("これは Excel ではありません")

        loader = DataLoader()
        result = loader.load(str(fake_xlsx))

        assert isinstance(result, LoadError)
        assert result.kind == "read_error"


class TestLoadPathNormalization:
    """pathlib.Path でパスが正規化される。"""

    def test_nonexistent_path_with_extra_slash(self) -> None:
        """余分なスラッシュが含まれるパスでも正しく処理する。"""
        loader = DataLoader()
        result = loader.load("//tmp//nonexistent//file.xlsx")

        assert isinstance(result, LoadError)
        assert result.kind == "file_not_found"


# 実際の Excel ファイルを使った統合テスト（ファイルが存在する場合のみ実行）
EXCEL_PATH = "/home/sagemaker-user/5cm-chart-analysis/data/20220916-koga-st-5cm-original-data.xlsx"


@pytest.mark.skipif(
    not os.path.exists(EXCEL_PATH),
    reason="サンプル Excel ファイルが存在しない場合はスキップ",
)
class TestLoadRealFile:
    """実際の Excel ファイルを読み込む統合テスト。

    Note:
        実際の Excel ファイルの列名が REQUIRED_COLUMNS と異なる場合
        （例: 全角「ＣＨ」vs 半角「CH」）、load() は LoadError を返すことがある。
        ここではファイルが読み込み可能であること（read_error にならない）のみ検証する。
    """

    def test_load_does_not_return_file_not_found_error(self) -> None:
        """ファイルが存在する場合は file_not_found エラーを返さない。"""
        loader = DataLoader()
        result = loader.load(EXCEL_PATH)

        if isinstance(result, LoadError):
            assert result.kind != "file_not_found", (
                "ファイルが存在するのに file_not_found エラーが返された"
            )
        # DataFrame または missing_columns エラーのいずれかが想定される結果

    def test_load_does_not_raise_exception(self) -> None:
        """実際のファイルを読み込んでも例外が送出されない。"""
        loader = DataLoader()
        # 例外が送出されればここでテスト失敗となる
        result = loader.load(EXCEL_PATH)
        assert result is not None

    def test_load_result_is_dataframe_or_load_error(self) -> None:
        """戻り値は DataFrame か LoadError のいずれかである。"""
        loader = DataLoader()
        result = loader.load(EXCEL_PATH)

        assert isinstance(result, (pd.DataFrame, LoadError)), (
            f"戻り値は DataFrame か LoadError であるべき。実際: {type(result)}"
        )


# ---------------------------------------------------------------------------
# 2.2: 必須列の存在検証処理
# ---------------------------------------------------------------------------


class TestValidateColumnsAllPresent:
    """全必須列が存在する場合は None を返す。"""

    def test_all_columns_present_returns_none(self) -> None:
        loader = DataLoader()
        df = make_valid_df()
        result = loader.validate_columns(df)

        assert result is None, "全列が存在する場合は None を返すべき"


class TestValidateColumnsMissing:
    """必須列が欠損する場合は LoadError(kind="missing_columns") を返す。"""

    def test_one_missing_column_returns_load_error(self) -> None:
        loader = DataLoader()
        df = make_valid_df().drop(columns=["キロ程"])
        result = loader.validate_columns(df)

        assert isinstance(result, LoadError)

    def test_one_missing_column_kind_is_missing_columns(self) -> None:
        loader = DataLoader()
        df = make_valid_df().drop(columns=["摩耗_測定値"])
        result = loader.validate_columns(df)

        assert isinstance(result, LoadError)
        assert result.kind == "missing_columns"

    def test_missing_columns_listed_in_result(self) -> None:
        loader = DataLoader()
        missing = ["キロ程", "CH"]
        df = make_valid_df().drop(columns=missing)
        result = loader.validate_columns(df)

        assert isinstance(result, LoadError)
        for col in missing:
            assert col in result.missing_columns, f"'{col}' が missing_columns に含まれるべき"

    def test_missing_columns_message_contains_column_name(self) -> None:
        loader = DataLoader()
        df = make_valid_df().drop(columns=["電柱番号"])
        result = loader.validate_columns(df)

        assert isinstance(result, LoadError)
        assert "電柱番号" in result.message

    def test_all_columns_missing(self) -> None:
        """全列欠損の場合も適切に処理する。"""
        loader = DataLoader()
        df = pd.DataFrame({"無関係列": [1, 2, 3]})
        result = loader.validate_columns(df)

        assert isinstance(result, LoadError)
        assert result.kind == "missing_columns"
        assert len(result.missing_columns) == len(REQUIRED_COLUMNS)

    def test_extra_columns_do_not_cause_error(self) -> None:
        """余分な列があっても必須列が揃っていればエラーなし。"""
        loader = DataLoader()
        df = make_valid_df()
        df["余分な列"] = 0
        result = loader.validate_columns(df)

        assert result is None


# ---------------------------------------------------------------------------
# 2.3: CH別グループ化機能
# ---------------------------------------------------------------------------


class TestGetChannelGroup:
    """CH 列の値でフィルタリングして個別 DataFrame を返す。"""

    def test_ch1_group_returns_correct_rows(self) -> None:
        loader = DataLoader()
        df = make_valid_df(rows=5)
        # CH 列: [1, 2, 3, 4, 1]
        result = loader.get_channel_group(df, ch=1)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2  # CH==1 は行0と行4
        assert (result["CH"] == 1).all()

    def test_ch2_group_returns_single_row(self) -> None:
        loader = DataLoader()
        df = make_valid_df(rows=5)
        result = loader.get_channel_group(df, ch=2)

        assert len(result) == 1
        assert (result["CH"] == 2).all()

    def test_returns_dataframe_type(self) -> None:
        loader = DataLoader()
        df = make_valid_df(rows=4)
        result = loader.get_channel_group(df, ch=3)

        assert isinstance(result, pd.DataFrame)

    def test_out_of_range_ch_returns_empty_dataframe(self) -> None:
        """CH 値が 1〜4 の範囲外の場合は空の DataFrame を返す。"""
        loader = DataLoader()
        df = make_valid_df(rows=5)
        result = loader.get_channel_group(df, ch=5)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_ch_zero_returns_empty_dataframe(self) -> None:
        loader = DataLoader()
        df = make_valid_df(rows=5)
        result = loader.get_channel_group(df, ch=0)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_ch_negative_returns_empty_dataframe(self) -> None:
        loader = DataLoader()
        df = make_valid_df(rows=5)
        result = loader.get_channel_group(df, ch=-1)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_out_of_range_ch_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """範囲外 CH のとき警告がログに出力される。"""
        loader = DataLoader()
        df = make_valid_df(rows=5)

        with caplog.at_level(logging.WARNING):
            loader.get_channel_group(df, ch=99)

        assert len(caplog.records) > 0, "範囲外 CH ではログ警告が出力されるべき"

    def test_original_row_order_is_preserved(self) -> None:
        """返却 DataFrame は元の行順を保持する。"""
        loader = DataLoader()
        df = make_valid_df(rows=5)
        result = loader.get_channel_group(df, ch=1)

        # 元の行0と行4が順序通り含まれることを確認
        expected_indices = df[df["CH"] == 1].index.tolist()
        assert result.index.tolist() == expected_indices


# ---------------------------------------------------------------------------
# LoadError データクラスの構造確認
# ---------------------------------------------------------------------------


class TestLoadErrorDataclass:
    """LoadError データクラスの構造と不変条件を検証する。"""

    def test_load_error_is_frozen(self) -> None:
        error = LoadError(kind="file_not_found", message="テスト")
        with pytest.raises(Exception):
            error.kind = "other"  # type: ignore[misc]

    def test_missing_columns_default_is_empty_list(self) -> None:
        error = LoadError(kind="file_not_found", message="テスト")
        assert error.missing_columns == []
        assert isinstance(error.missing_columns, list)

    def test_missing_columns_can_be_specified(self) -> None:
        cols = ["キロ程", "CH"]
        error = LoadError(kind="missing_columns", message="欠損あり", missing_columns=cols)
        assert error.missing_columns == cols
