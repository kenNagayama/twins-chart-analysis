"""ParameterValidator のユニットテスト (タスク 3.1 / 3.2)"""
from __future__ import annotations

import pytest

from src.parameter_validator import (
    InvalidThresholdError,
    InvalidWindowError,
    ParameterValidator,
)


# ---------------------------------------------------------------------------
# 3.1: ウィンドウ幅バリデーションの実装
# ---------------------------------------------------------------------------


class TestValidateWindowValid:
    """有効なウィンドウ幅は検証済み値をそのまま返す。"""

    def test_valid_window_returns_int(self) -> None:
        validator = ParameterValidator()
        result = validator.validate_window(window_size=10, data_length=100)

        assert isinstance(result, int)

    def test_valid_window_returns_same_value(self) -> None:
        validator = ParameterValidator()
        result = validator.validate_window(window_size=10, data_length=100)

        assert result == 10

    def test_minimum_valid_window(self) -> None:
        """最小有効値 window_size=1 は通過する。"""
        validator = ParameterValidator()
        result = validator.validate_window(window_size=1, data_length=100)

        assert result == 1

    def test_maximum_valid_window(self) -> None:
        """window_size == data_length は有効範囲内。"""
        validator = ParameterValidator()
        result = validator.validate_window(window_size=100, data_length=100)

        assert result == 100

    def test_window_equal_to_data_length_is_valid(self) -> None:
        validator = ParameterValidator()
        result = validator.validate_window(window_size=50, data_length=50)

        assert isinstance(result, int)
        assert result == 50


class TestValidateWindowInvalid:
    """無効なウィンドウ幅は InvalidWindowError を返す。"""

    def test_zero_window_returns_error(self) -> None:
        validator = ParameterValidator()
        result = validator.validate_window(window_size=0, data_length=100)

        assert isinstance(result, InvalidWindowError)

    def test_negative_window_returns_error(self) -> None:
        validator = ParameterValidator()
        result = validator.validate_window(window_size=-5, data_length=100)

        assert isinstance(result, InvalidWindowError)

    def test_window_exceeds_data_length_returns_error(self) -> None:
        validator = ParameterValidator()
        result = validator.validate_window(window_size=101, data_length=100)

        assert isinstance(result, InvalidWindowError)

    def test_error_contains_window_size(self) -> None:
        validator = ParameterValidator()
        result = validator.validate_window(window_size=0, data_length=100)

        assert isinstance(result, InvalidWindowError)
        assert result.window_size == 0

    def test_error_contains_data_length(self) -> None:
        validator = ParameterValidator()
        result = validator.validate_window(window_size=200, data_length=100)

        assert isinstance(result, InvalidWindowError)
        assert result.data_length == 100

    def test_error_message_contains_valid_range(self) -> None:
        """エラーメッセージに有効範囲（1 以上 data_length 以下）が含まれる。"""
        validator = ParameterValidator()
        result = validator.validate_window(window_size=0, data_length=100)

        assert isinstance(result, InvalidWindowError)
        assert "1" in result.message
        assert "100" in result.message

    def test_negative_window_error_has_data_length(self) -> None:
        validator = ParameterValidator()
        result = validator.validate_window(window_size=-1, data_length=50)

        assert isinstance(result, InvalidWindowError)
        assert result.data_length == 50


class TestInvalidWindowErrorDataclass:
    """InvalidWindowError データクラスの構造を確認する。"""

    def test_is_frozen(self) -> None:
        error = InvalidWindowError(window_size=0, data_length=100, message="テスト")
        with pytest.raises(Exception):
            error.window_size = 1  # type: ignore[misc]

    def test_fields_accessible(self) -> None:
        error = InvalidWindowError(window_size=-1, data_length=50, message="無効")
        assert error.window_size == -1
        assert error.data_length == 50
        assert error.message == "無効"


# ---------------------------------------------------------------------------
# 3.1: ensure_odd_window（奇数補正）の実装
# ---------------------------------------------------------------------------


class TestEnsureOddWindow:
    """偶数ウィンドウ幅は +1 して奇数に補正する。"""

    def test_even_window_becomes_odd(self) -> None:
        validator = ParameterValidator()
        result = validator.ensure_odd_window(4)

        assert result == 5

    def test_odd_window_unchanged(self) -> None:
        validator = ParameterValidator()
        result = validator.ensure_odd_window(5)

        assert result == 5

    def test_minimum_value_1_unchanged(self) -> None:
        """最小値 1（奇数）はそのまま返す。"""
        validator = ParameterValidator()
        result = validator.ensure_odd_window(1)

        assert result == 1

    def test_value_2_becomes_3(self) -> None:
        validator = ParameterValidator()
        result = validator.ensure_odd_window(2)

        assert result == 3

    def test_large_even_value(self) -> None:
        validator = ParameterValidator()
        result = validator.ensure_odd_window(100)

        assert result == 101

    def test_large_odd_value(self) -> None:
        validator = ParameterValidator()
        result = validator.ensure_odd_window(99)

        assert result == 99

    def test_result_is_always_odd(self) -> None:
        """任意の入力値に対して結果が奇数であることを検証する。"""
        validator = ParameterValidator()
        for w in range(1, 20):
            result = validator.ensure_odd_window(w)
            assert result % 2 == 1, f"window={w} の補正結果 {result} が奇数でない"


# ---------------------------------------------------------------------------
# 3.2: Z-Score 閾値バリデーションの実装
# ---------------------------------------------------------------------------


class TestValidateThresholdValid:
    """有効な Z-Score 閾値（正の実数）は検証済み値を返す。"""

    def test_positive_float_returns_float(self) -> None:
        validator = ParameterValidator()
        result = validator.validate_threshold(2.5)

        assert isinstance(result, float)

    def test_positive_float_returns_same_value(self) -> None:
        validator = ParameterValidator()
        result = validator.validate_threshold(3.0)

        assert result == 3.0

    def test_small_positive_value(self) -> None:
        validator = ParameterValidator()
        result = validator.validate_threshold(0.1)

        assert result == pytest.approx(0.1)

    def test_integer_as_float(self) -> None:
        """int として渡された正値も float として受け付ける。"""
        validator = ParameterValidator()
        result = validator.validate_threshold(2)

        assert isinstance(result, (int, float))
        assert result == 2

    def test_large_threshold_is_valid(self) -> None:
        validator = ParameterValidator()
        result = validator.validate_threshold(100.0)

        assert result == 100.0


class TestValidateThresholdInvalid:
    """無効な Z-Score 閾値は InvalidThresholdError を返す。"""

    def test_zero_returns_error(self) -> None:
        validator = ParameterValidator()
        result = validator.validate_threshold(0.0)

        assert isinstance(result, InvalidThresholdError)

    def test_negative_value_returns_error(self) -> None:
        validator = ParameterValidator()
        result = validator.validate_threshold(-1.0)

        assert isinstance(result, InvalidThresholdError)

    def test_negative_small_value_returns_error(self) -> None:
        validator = ParameterValidator()
        result = validator.validate_threshold(-0.001)

        assert isinstance(result, InvalidThresholdError)

    def test_error_contains_threshold_value(self) -> None:
        validator = ParameterValidator()
        result = validator.validate_threshold(-2.0)

        assert isinstance(result, InvalidThresholdError)
        assert result.threshold == -2.0

    def test_error_message_guides_user(self) -> None:
        """エラーメッセージに正の実数を指定するよう案内が含まれる。"""
        validator = ParameterValidator()
        result = validator.validate_threshold(0.0)

        assert isinstance(result, InvalidThresholdError)
        # メッセージが存在し、何らかの案内が含まれること
        assert len(result.message) > 0

    def test_zero_integer_returns_error(self) -> None:
        validator = ParameterValidator()
        result = validator.validate_threshold(0)

        assert isinstance(result, InvalidThresholdError)


class TestInvalidThresholdErrorDataclass:
    """InvalidThresholdError データクラスの構造を確認する。"""

    def test_is_frozen(self) -> None:
        error = InvalidThresholdError(threshold=0.0, message="テスト")
        with pytest.raises(Exception):
            error.threshold = 1.0  # type: ignore[misc]

    def test_fields_accessible(self) -> None:
        error = InvalidThresholdError(threshold=-1.5, message="負の値は無効")
        assert error.threshold == -1.5
        assert error.message == "負の値は無効"
