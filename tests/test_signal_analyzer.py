"""SignalAnalyzer のユニットテスト (タスク 5.1 / 5.2 / 5.3)"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.signal_analyzer import FFTResult, RMSResult, STFTResult, SignalAnalyzer


# ---------------------------------------------------------------------------
# フィクスチャ
# ---------------------------------------------------------------------------


def make_wear_series(n: int = 100, seed: int = 42) -> pd.Series:
    """テスト用の摩耗値 Series を生成するヘルパー。"""
    rng = np.random.default_rng(seed)
    base = np.linspace(40.0, 38.0, n)
    noise = rng.normal(0, 0.5, n)
    return pd.Series(base + noise, name="摩耗_測定値")


def make_constant_series(value: float = 5.0, n: int = 50) -> pd.Series:
    """均一な値を持つ Series を生成するヘルパー。"""
    return pd.Series([value] * n, name="テスト")


# ---------------------------------------------------------------------------
# 5.1: スライドウィンドウ RMS の実装
# ---------------------------------------------------------------------------


class TestComputeRMS:
    """スライドウィンドウ RMS（compute_rms）の動作検証。"""

    def test_returns_rms_result(self) -> None:
        """戻り値が RMSResult であることを確認する。"""
        sa = SignalAnalyzer()
        s = make_wear_series()
        result = sa.compute_rms(s, window_size=10)
        assert isinstance(result, RMSResult)

    def test_rms_series_length(self) -> None:
        """rms_series の長さが入力 Series と等しい。"""
        sa = SignalAnalyzer()
        s = make_wear_series(n=100)
        result = sa.compute_rms(s, window_size=10)
        assert len(result.rms_series) == len(s)

    def test_rms_series_is_pandas_series(self) -> None:
        """rms_series が pd.Series であることを確認する。"""
        sa = SignalAnalyzer()
        s = make_wear_series()
        result = sa.compute_rms(s, window_size=5)
        assert isinstance(result.rms_series, pd.Series)

    def test_rms_non_negative(self) -> None:
        """RMS 値はすべて非負（NaN を除く）。"""
        sa = SignalAnalyzer()
        s = make_wear_series(n=50)
        result = sa.compute_rms(s, window_size=5)
        valid_values = result.rms_series.dropna()
        assert (valid_values >= 0).all(), "RMS 値は非負であるべき"

    def test_rms_of_constant_equals_constant(self) -> None:
        """均一な値の Series の RMS は元の値に等しい。"""
        sa = SignalAnalyzer()
        value = 5.0
        s = make_constant_series(value=value, n=50)
        result = sa.compute_rms(s, window_size=5)
        valid_values = result.rms_series.dropna()
        np.testing.assert_allclose(valid_values.values, value, rtol=1e-10)

    def test_rms_nan_at_start(self) -> None:
        """ウィンドウ幅未満の先頭区間は NaN になる（min_periods=window_size のデフォルト動作）。"""
        sa = SignalAnalyzer()
        n = 50
        window_size = 10
        s = make_wear_series(n=n)
        result = sa.compute_rms(s, window_size=window_size)
        # 先頭 window_size - 1 点が NaN
        assert result.rms_series.iloc[: window_size - 1].isna().all()
        # window_size 点目以降は NaN でない
        assert result.rms_series.iloc[window_size - 1 :].notna().all()

    def test_rms_index_preserved(self) -> None:
        """入力 Series のインデックスが保持される。"""
        sa = SignalAnalyzer()
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=[10, 20, 30, 40, 50])
        result = sa.compute_rms(s, window_size=3)
        assert list(result.rms_series.index) == [10, 20, 30, 40, 50]

    def test_rms_window_1_returns_absolute_values(self) -> None:
        """window_size=1 の場合は各点の絶対値が返る。"""
        sa = SignalAnalyzer()
        s = pd.Series([3.0, -4.0, 5.0])
        result = sa.compute_rms(s, window_size=1)
        expected = np.array([3.0, 4.0, 5.0])
        np.testing.assert_allclose(result.rms_series.values, expected)


# ---------------------------------------------------------------------------
# 5.2: FFT スペクトル算出の実装
# ---------------------------------------------------------------------------


class TestComputeFFT:
    """FFT スペクトル算出（compute_fft）の動作検証。"""

    def test_returns_fft_result(self) -> None:
        """戻り値が FFTResult であることを確認する。"""
        sa = SignalAnalyzer()
        s = make_wear_series(n=100)
        result = sa.compute_fft(s, window_size=64)
        assert isinstance(result, FFTResult)

    def test_frequencies_length(self) -> None:
        """frequencies の長さは window_size // 2 + 1 である（rfft の仕様）。"""
        sa = SignalAnalyzer()
        s = make_wear_series(n=100)
        window_size = 64
        result = sa.compute_fft(s, window_size=window_size)
        expected_len = window_size // 2 + 1
        assert len(result.frequencies) == expected_len

    def test_amplitudes_length_matches_frequencies(self) -> None:
        """amplitudes の長さは frequencies と等しい。"""
        sa = SignalAnalyzer()
        s = make_wear_series(n=100)
        result = sa.compute_fft(s, window_size=64)
        assert len(result.amplitudes) == len(result.frequencies)

    def test_frequencies_non_negative(self) -> None:
        """rfft の周波数軸は非負（0 以上）。"""
        sa = SignalAnalyzer()
        s = make_wear_series(n=100)
        result = sa.compute_fft(s, window_size=64)
        assert (result.frequencies >= 0).all(), "rfft の周波数は非負であるべき"

    def test_amplitudes_non_negative(self) -> None:
        """振幅スペクトルは非負（絶対値）。"""
        sa = SignalAnalyzer()
        s = make_wear_series(n=100)
        result = sa.compute_fft(s, window_size=64)
        assert (result.amplitudes >= 0).all(), "振幅スペクトルは非負であるべき"

    def test_frequency_spacing(self) -> None:
        """周波数間隔は 1 / (window_size * 0.05) m^-1 である。"""
        sa = SignalAnalyzer()
        s = make_wear_series(n=200)
        window_size = 100
        result = sa.compute_fft(s, window_size=window_size)
        expected_df = 1.0 / (window_size * SignalAnalyzer.SAMPLE_SPACING_M)
        if len(result.frequencies) > 1:
            actual_df = result.frequencies[1] - result.frequencies[0]
            np.testing.assert_allclose(actual_df, expected_df, rtol=1e-10)

    def test_dc_component_first(self) -> None:
        """frequencies[0] が 0（DC 成分）である。"""
        sa = SignalAnalyzer()
        s = make_wear_series(n=100)
        result = sa.compute_fft(s, window_size=64)
        assert result.frequencies[0] == pytest.approx(0.0)

    def test_frequencies_numpy_array(self) -> None:
        """frequencies が np.ndarray であることを確認する。"""
        sa = SignalAnalyzer()
        s = make_wear_series(n=100)
        result = sa.compute_fft(s, window_size=64)
        assert isinstance(result.frequencies, np.ndarray)

    def test_amplitudes_numpy_array(self) -> None:
        """amplitudes が np.ndarray であることを確認する。"""
        sa = SignalAnalyzer()
        s = make_wear_series(n=100)
        result = sa.compute_fft(s, window_size=64)
        assert isinstance(result.amplitudes, np.ndarray)

    def test_window_size_determines_output_length(self) -> None:
        """window_size が FFT 出力長を決定する。異なる window_size で異なる出力長。"""
        sa = SignalAnalyzer()
        s = make_wear_series(n=200)
        result_32 = sa.compute_fft(s, window_size=32)
        result_64 = sa.compute_fft(s, window_size=64)
        assert len(result_32.frequencies) == 32 // 2 + 1
        assert len(result_64.frequencies) == 64 // 2 + 1


# ---------------------------------------------------------------------------
# 5.3: STFT スペクトログラム算出の実装
# ---------------------------------------------------------------------------


class TestComputeSTFT:
    """STFT スペクトログラム算出（compute_stft）の動作検証。"""

    def test_returns_stft_result(self) -> None:
        """戻り値が STFTResult であることを確認する。"""
        sa = SignalAnalyzer()
        s = make_wear_series(n=200)
        result = sa.compute_stft(s, window_size=32)
        assert isinstance(result, STFTResult)

    def test_frequencies_length(self) -> None:
        """frequencies の長さは window_size // 2 + 1 である（stft の仕様）。"""
        sa = SignalAnalyzer()
        s = make_wear_series(n=200)
        window_size = 32
        result = sa.compute_stft(s, window_size=window_size)
        expected_len = window_size // 2 + 1
        assert len(result.frequencies) == expected_len

    def test_spectrogram_shape(self) -> None:
        """spectrogram の形状は [len(frequencies), len(positions)] である。"""
        sa = SignalAnalyzer()
        s = make_wear_series(n=200)
        window_size = 32
        result = sa.compute_stft(s, window_size=window_size)
        assert result.spectrogram.shape == (len(result.frequencies), len(result.positions))

    def test_spectrogram_non_negative(self) -> None:
        """spectrogram の値は非負（絶対値）。"""
        sa = SignalAnalyzer()
        s = make_wear_series(n=200)
        result = sa.compute_stft(s, window_size=32)
        assert (result.spectrogram >= 0).all(), "スペクトログラム値は非負であるべき"

    def test_frequencies_non_negative(self) -> None:
        """STFT の周波数軸は非負。"""
        sa = SignalAnalyzer()
        s = make_wear_series(n=200)
        result = sa.compute_stft(s, window_size=32)
        assert (result.frequencies >= 0).all(), "周波数は非負であるべき"

    def test_frequencies_numpy_array(self) -> None:
        """frequencies が np.ndarray であることを確認する。"""
        sa = SignalAnalyzer()
        s = make_wear_series(n=200)
        result = sa.compute_stft(s, window_size=32)
        assert isinstance(result.frequencies, np.ndarray)

    def test_positions_numpy_array(self) -> None:
        """positions が np.ndarray であることを確認する。"""
        sa = SignalAnalyzer()
        s = make_wear_series(n=200)
        result = sa.compute_stft(s, window_size=32)
        assert isinstance(result.positions, np.ndarray)

    def test_spectrogram_numpy_array(self) -> None:
        """spectrogram が np.ndarray であることを確認する。"""
        sa = SignalAnalyzer()
        s = make_wear_series(n=200)
        result = sa.compute_stft(s, window_size=32)
        assert isinstance(result.spectrogram, np.ndarray)

    def test_spectrogram_is_2d(self) -> None:
        """spectrogram が 2D 配列である。"""
        sa = SignalAnalyzer()
        s = make_wear_series(n=200)
        result = sa.compute_stft(s, window_size=32)
        assert result.spectrogram.ndim == 2

    def test_dc_component_at_zero_frequency(self) -> None:
        """frequencies[0] が 0（DC 成分）。"""
        sa = SignalAnalyzer()
        s = make_wear_series(n=200)
        result = sa.compute_stft(s, window_size=32)
        assert result.frequencies[0] == pytest.approx(0.0)

    def test_positions_are_non_negative(self) -> None:
        """positions は非負の値（位置情報）。"""
        sa = SignalAnalyzer()
        s = make_wear_series(n=200)
        result = sa.compute_stft(s, window_size=32)
        assert (result.positions >= 0).all()


# ---------------------------------------------------------------------------
# データクラスの構造確認
# ---------------------------------------------------------------------------


class TestRMSResultDataclass:
    """RMSResult データクラスの構造を検証する。"""

    def test_is_frozen(self) -> None:
        """frozen=True により属性の再代入が阻止される。"""
        s = pd.Series([1.0, 2.0, 3.0])
        result = RMSResult(rms_series=s)
        with pytest.raises(Exception):
            result.rms_series = pd.Series([4.0])  # type: ignore[misc]

    def test_fields_accessible(self) -> None:
        """全フィールドにアクセスできる。"""
        s = pd.Series([1.0, 2.0, 3.0])
        result = RMSResult(rms_series=s)
        pd.testing.assert_series_equal(result.rms_series, s)


class TestFFTResultDataclass:
    """FFTResult データクラスの構造を検証する。"""

    def test_is_frozen(self) -> None:
        """frozen=True により属性の再代入が阻止される。"""
        freq = np.array([0.0, 1.0, 2.0])
        amp = np.array([1.0, 0.5, 0.2])
        result = FFTResult(frequencies=freq, amplitudes=amp)
        with pytest.raises(Exception):
            result.frequencies = np.array([3.0])  # type: ignore[misc]

    def test_fields_accessible(self) -> None:
        """全フィールドにアクセスできる。"""
        freq = np.array([0.0, 1.0, 2.0])
        amp = np.array([1.0, 0.5, 0.2])
        result = FFTResult(frequencies=freq, amplitudes=amp)
        np.testing.assert_array_equal(result.frequencies, freq)
        np.testing.assert_array_equal(result.amplitudes, amp)


class TestSTFTResultDataclass:
    """STFTResult データクラスの構造を検証する。"""

    def test_is_frozen(self) -> None:
        """frozen=True により属性の再代入が阻止される。"""
        freq = np.array([0.0, 1.0])
        pos = np.array([0.0, 1.0, 2.0])
        spec = np.ones((2, 3))
        result = STFTResult(frequencies=freq, positions=pos, spectrogram=spec)
        with pytest.raises(Exception):
            result.frequencies = np.array([3.0])  # type: ignore[misc]

    def test_fields_accessible(self) -> None:
        """全フィールドにアクセスできる。"""
        freq = np.array([0.0, 1.0])
        pos = np.array([0.0, 1.0, 2.0])
        spec = np.ones((2, 3))
        result = STFTResult(frequencies=freq, positions=pos, spectrogram=spec)
        np.testing.assert_array_equal(result.frequencies, freq)
        np.testing.assert_array_equal(result.positions, pos)
        np.testing.assert_array_equal(result.spectrogram, spec)
