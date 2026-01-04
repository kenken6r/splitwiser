import pandas as pd
import streamlit as st

st.set_page_config(page_title="裏", layout="wide")

APP_PASSWORD = "tet1213"

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    # 文字化けする場合は encoding="utf-8-sig" を試してください
    df = pd.read_csv(path)

    required_cols = [
        "氏名", "かな", "入省年次", "出身大学", "出身高校",
        "年度", "年目", "ポスト"
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSVに必要な列が足りません: {missing}")

    # 年度を並べ替えに使える形へ（4桁西暦を抽出）
    y = df["年度"].astype(str).str.extract(r"(\d{4})")[0]
    df["年度_num"] = pd.to_numeric(y, errors="coerce")

    return df


def auth_gate() -> None:
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return

    st.title("認証が必要です")
    pw = st.text_input("パスワードを入力してください", type="password")

    col1, col2 = st.columns([1, 4])
    with col1:
        login_clicked = st.button("送信")

    if login_clicked:
        if pw == APP_PASSWORD:
            st.session_state.authenticated = True
            st.success("認証できました。")
            st.rerun()
        else:
            st.error("パスワードが違います。")

    st.stop()


# まず認証
auth_gate()

# 認証後の画面
st.title("裏")

df = load_data("df_long.csv")

name_input = st.text_input("氏名を入力", placeholder="例: 山田 太郎")

candidate_names = []
if name_input:
    candidate_names = (
        df.loc[df["氏名"].astype(str).str.contains(name_input, na=False), "氏名"]
        .dropna()
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

if name_input:
    st.write(f"ヒット件数：{len(candidate_names)}")
    st.write("候補者名：")
    if candidate_names:
        st.write(" / ".join(candidate_names))
    else:
        st.write("該当なし")

selected_name = None
if candidate_names:
    selected_name = st.selectbox("候補から選択", candidate_names)
elif name_input:
    if name_input in set(df["氏名"].dropna().astype(str)):
        selected_name = name_input

if not selected_name:
    st.info("氏名を入力してください。")
    st.stop()

person_df = df[df["氏名"].astype(str) == str(selected_name)].copy()
if person_df.empty:
    st.warning("該当者が見つかりませんでした。表記ゆれがないか確認してください。")
    st.stop()

if person_df["年度_num"].notna().any():
    latest_idx = person_df["年度_num"].idxmax()
else:
    latest_idx = person_df.index.max()

latest_row = person_df.loc[latest_idx]

st.subheader("基本情報")

col1, col2, col3, col4, col5 = st.columns(5)
col1.write("氏名")
col1.write(latest_row.get("氏名", ""))

col2.write("かな")
col2.write(latest_row.get("かな", ""))

col3.write("入省年次")
col3.write(latest_row.get("入省年次", ""))

col4.write("出身大学")
col4.write(latest_row.get("出身大学", ""))

col5.write("出身高校")
col5.write(latest_row.get("出身高校", ""))

st.subheader("ポスト遍歴")

history = person_df[["年度", "年目", "ポスト", "年度_num"]].copy()

if history["年度_num"].notna().any():
    history = history.sort_values(["年度_num", "年度"], ascending=[True, True])
else:
    history = history.sort_index()

history = history.drop(columns=["年度_num"])

st.dataframe(history, use_container_width=True, hide_index=True)

# 任意: ログアウトボタン
st.divider()
if st.button("ログアウト"):
    st.session_state.authenticated = False
    st.rerun()