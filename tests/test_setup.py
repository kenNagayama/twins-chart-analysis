"""タスク1: プロジェクト環境セットアップの検証テスト"""
from __future__ import annotations

import importlib
from pathlib import Path


# プロジェクトルートパス
PROJECT_ROOT = Path(__file__).parent.parent


class TestTask11DependencyPackages:
    """1.1 依存パッケージの定義と仮想環境構築"""

    def test_pyproject_toml_exists(self):
        """pyproject.toml が存在すること"""
        assert (PROJECT_ROOT / "pyproject.toml").exists()

    def test_pandas_importable(self):
        """pandas がインポートできること"""
        import pandas as pd
        assert pd.__version__ >= "2.0.0"

    def test_numpy_importable(self):
        """NumPy がインポートできること"""
        import numpy as np
        assert np.__version__ >= "1.26.0"

    def test_scipy_importable(self):
        """SciPy がインポートできること"""
        import scipy
        assert scipy.__version__ >= "1.11.0"

    def test_plotly_importable(self):
        """Plotly がインポートできること"""
        import plotly
        assert plotly.__version__ >= "5.0.0"

    def test_streamlit_importable(self):
        """Streamlit がインポートできること"""
        import streamlit
        assert streamlit.__version__ >= "1.28.0"

    def test_openpyxl_importable(self):
        """openpyxl がインポートできること"""
        import openpyxl
        assert openpyxl.__version__ >= "3.1.0"


class TestTask12DirectoryStructure:
    """1.2 ディレクトリ構造とエントリポイントの骨格作成"""

    def test_src_directory_exists(self):
        """`src/` ディレクトリが存在すること"""
        assert (PROJECT_ROOT / "src").is_dir()

    def test_output_directory_exists(self):
        """`output/` ディレクトリが存在すること"""
        assert (PROJECT_ROOT / "output").is_dir()

    def test_data_loader_module_exists(self):
        """`src/data_loader.py` が存在すること"""
        assert (PROJECT_ROOT / "src" / "data_loader.py").exists()

    def test_noise_filter_module_exists(self):
        """`src/noise_filter.py` が存在すること"""
        assert (PROJECT_ROOT / "src" / "noise_filter.py").exists()

    def test_signal_analyzer_module_exists(self):
        """`src/signal_analyzer.py` が存在すること"""
        assert (PROJECT_ROOT / "src" / "signal_analyzer.py").exists()

    def test_anomaly_detector_module_exists(self):
        """`src/anomaly_detector.py` が存在すること"""
        assert (PROJECT_ROOT / "src" / "anomaly_detector.py").exists()

    def test_visualizer_module_exists(self):
        """`src/visualizer.py` が存在すること"""
        assert (PROJECT_ROOT / "src" / "visualizer.py").exists()

    def test_app_py_exists(self):
        """`app.py` が存在すること"""
        assert (PROJECT_ROOT / "app.py").exists()

    def test_src_modules_importable(self):
        """src/ 配下のモジュールがインポートできること"""
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))
        for module_name in [
            "src.data_loader",
            "src.noise_filter",
            "src.signal_analyzer",
            "src.anomaly_detector",
            "src.visualizer",
        ]:
            mod = importlib.import_module(module_name)
            assert mod is not None, f"{module_name} がインポートできない"
