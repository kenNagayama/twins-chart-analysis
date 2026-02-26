"""Streamlit アプリのエントリポイント"""
from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="トロリ線摩耗 EDA ツール",
    page_icon="📊",
    layout="wide",
)

st.title("トロリ線摩耗検測データ 探索的データ分析（EDA）ツール")
st.write("サイドバーからパラメータを設定し、分析を実行してください。")
