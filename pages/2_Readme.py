# pages/2_Readme.py
import streamlit as st

# Page config
st.set_page_config(page_title="Readme", layout="wide")

st.title("Readme")

# Language tabs (default: English)
tab_en, tab_ja = st.tabs(["English", "日本語"])

with tab_en:
    st.markdown(
        """
### What this app does
- Create a page with a main currency and an optional sub currency
- Add members and transactions
- Compute balances automatically
- Convert balances using an FX rate

### Notes
- This is a lightweight Splitwise style tool.
- Data is stored locally by the app.
- If you delete or restore members, you may need to update existing transactions
  to keep balances consistent.

### How to use
1. Create a page (or open an existing one)
2. Add members and transactions
3. Check balances and settle up
"""
    )

with tab_ja:
    st.markdown(
        """
### このアプリについて
- メイン通貨とサブ通貨を設定したページを作成できます
- メンバーと取引を追加できます
- 残高は自動で計算されます
- 為替レートを使って通貨換算ができます

### 注意
- このアプリは簡易的な Splitwise 風ツールです
- データはアプリ内に保存されます
- **メンバーを削除・復元した場合、取引データを更新しないと
  残高が正しく計算されないことがあります**

### 使い方
1. ページを作成する（または既存ページを開く）
2. メンバーと取引を追加する
3. 残高を確認して精算する
"""
    )

# Back to main
if st.button("Back to Main"):
    st.switch_page("main.py")