"""Processing Layer: パラメータ検証"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InvalidWindowError:
    """ウィンドウ幅バリデーションエラー値オブジェクト。

    Attributes:
        window_size: 検証に失敗したウィンドウ幅
        data_length: データの長さ（有効範囲の上限）
        message: ユーザ向けエラーメッセージ（有効範囲を含む）
    """

    window_size: int
    data_length: int
    message: str


@dataclass(frozen=True)
class InvalidThresholdError:
    """Z-Score 閾値バリデーションエラー値オブジェクト。

    Attributes:
        threshold: 検証に失敗した閾値
        message: ユーザ向けエラーメッセージ
    """

    threshold: float
    message: str


@dataclass(frozen=True)
class ValidatedParams:
    """バリデーション済みパラメータ値オブジェクト。

    Attributes:
        window_size: バリデーション済みウィンドウ幅（奇数補正済み）
        zscore_threshold: バリデーション済み Z-Score 閾値
        polyorder: SG フィルタ多項式次数（デフォルト 2）
    """

    window_size: int
    zscore_threshold: float
    polyorder: int = 2


class ParameterValidator:
    """ウィンドウ幅・閾値等のユーザ入力パラメータを一元的に検証するクラス。

    エラー時は例外を送出せず、エラー値オブジェクトを返す。
    """

    def validate_window(
        self,
        window_size: int,
        data_length: int,
    ) -> int | InvalidWindowError:
        """ウィンドウ幅を検証し、有効であれば検証済み値を返す。

        有効条件: 1 <= window_size <= data_length

        Args:
            window_size: 検証するウィンドウ幅
            data_length: データの長さ（有効範囲の上限）

        Returns:
            有効な場合は window_size（int）を返す。
            無効な場合は InvalidWindowError を返す（例外を送出しない）。
        """
        if window_size < 1 or window_size > data_length:
            return InvalidWindowError(
                window_size=window_size,
                data_length=data_length,
                message=(
                    f"ウィンドウ幅 {window_size} は有効範囲外です。"
                    f"1 以上 {data_length} 以下の整数を指定してください。"
                ),
            )
        return window_size

    def validate_threshold(
        self,
        threshold: float,
    ) -> float | InvalidThresholdError:
        """Z-Score 閾値を検証し、有効であれば検証済み値を返す。

        有効条件: threshold > 0（正の実数）

        Args:
            threshold: 検証する Z-Score 閾値

        Returns:
            有効な場合は threshold（float）を返す。
            無効な場合は InvalidThresholdError を返す（例外を送出しない）。
        """
        if threshold <= 0:
            return InvalidThresholdError(
                threshold=float(threshold),
                message=(
                    f"Z-Score 閾値 {threshold} は無効です。"
                    "正の実数（例: 2.5）を指定してください。"
                ),
            )
        return float(threshold)

    def ensure_odd_window(self, window_size: int) -> int:
        """ウィンドウ幅が偶数の場合は +1 して奇数に補正して返す。

        Savitzky-Golay フィルタは奇数のウィンドウ幅を必要とするため、
        偶数の場合は自動的に +1 補正する。

        Args:
            window_size: 補正対象のウィンドウ幅

        Returns:
            奇数のウィンドウ幅。元が奇数の場合はそのまま、偶数の場合は +1 した値。
        """
        if window_size % 2 == 0:
            return window_size + 1
        return window_size
