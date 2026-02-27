"""ProcessingLayer: RMS・FFT・STFT"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import stft


@dataclass(frozen=True)
class RMSResult:
    """RMS 算出結果値オブジェクト。

    Attributes:
        rms_series: ウィンドウごとの RMS 値（ウィンドウ未満の先頭区間は NaN）
    """

    rms_series: pd.Series


@dataclass(frozen=True)
class FFTResult:
    """FFT 振幅スペクトル算出結果値オブジェクト。

    Attributes:
        frequencies: 周波数軸 (m^-1)
        amplitudes: 振幅スペクトル
    """

    frequencies: np.ndarray
    amplitudes: np.ndarray


@dataclass(frozen=True)
class STFTResult:
    """STFT スペクトログラム算出結果値オブジェクト。

    Attributes:
        frequencies: 周波数軸 (m^-1)
        positions: キロ程軸 (m)
        spectrogram: 2D 振幅スペクトル [freq x position]
    """

    frequencies: np.ndarray
    positions: np.ndarray
    spectrogram: np.ndarray


class SignalAnalyzer:
    """RMS・FFT・STFT 信号解析を実行するクラス。

    入力は ParameterValidator で検証済みのウィンドウ幅を受け取る。
    5cm = 0.05m のサンプル間隔を前提とする。
    状態を持たず、純粋関数的に動作する。
    """

    SAMPLE_SPACING_M: float = 0.05  # 5cm ピッチ

    def compute_rms(
        self, series: pd.Series, window_size: int
    ) -> RMSResult:
        """スライドウィンドウ RMS を算出する。

        pandas rolling の apply を用いて sqrt(mean(x**2)) を各ウィンドウに適用する。
        ウィンドウが満たない先頭区間は NaN となる（min_periods=window_size のデフォルト動作）。

        Args:
            series: 摩耗値 Series（フィルタリング済み）
            window_size: ウィンドウ幅（ParameterValidator で検証済み）

        Returns:
            RMSResult（rms_series: ウィンドウごとの RMS 値）
        """
        rms_values = series.rolling(window_size).apply(
            lambda x: np.sqrt(np.mean(x ** 2)), raw=True
        )
        return RMSResult(rms_series=rms_values)

    def compute_fft(
        self, series: pd.Series, window_size: int
    ) -> FFTResult:
        """指定ウィンドウ長の FFT 振幅スペクトルを算出する。

        numpy.fft.rfft を使い、最初の window_size 点のデータで FFT を計算する。
        周波数軸は rfftfreq(n, d=0.05) で計算する（5cm = 0.05m ピッチ）。

        Args:
            series: 摩耗値 Series（フィルタリング済み）
            window_size: FFT ウィンドウ幅（ParameterValidator で検証済み）

        Returns:
            FFTResult（frequencies: m^-1、amplitudes: 振幅スペクトル）
        """
        n = window_size
        data = series.to_numpy()[:n]
        spectrum = np.fft.rfft(data)
        frequencies = np.fft.rfftfreq(n, d=self.SAMPLE_SPACING_M)
        amplitudes = np.abs(spectrum)
        return FFTResult(frequencies=frequencies, amplitudes=amplitudes)

    def compute_stft(
        self,
        series: pd.Series,
        window_size: int,
    ) -> STFTResult:
        """STFT スペクトログラムを算出する。

        scipy.signal.stft を nperseg=window_size として呼び出し、
        時間-周波数スペクトルを算出する。
        noverlap=window_size//2 を基本設定とする。
        fs=1/SAMPLE_SPACING_M（20.0 samples/m）を設定し、
        周波数軸は m^-1、位置軸は m 単位で返す。

        Args:
            series: 摩耗値 Series（フィルタリング済み）
            window_size: STFT ウィンドウ幅（ParameterValidator で検証済み）

        Returns:
            STFTResult（frequencies: m^-1、positions: m、spectrogram: 2D 振幅 [freq x pos]）
        """
        fs = 1.0 / self.SAMPLE_SPACING_M  # 20.0 samples/m
        noverlap = window_size // 2
        frequencies, positions, zxx = stft(
            series.to_numpy(),
            fs=fs,
            nperseg=window_size,
            noverlap=noverlap,
        )
        spectrogram = np.abs(zxx)
        return STFTResult(
            frequencies=frequencies,
            positions=positions,
            spectrogram=spectrogram,
        )
