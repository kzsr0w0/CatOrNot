import streamlit as st
import pandas as pd
import requests

st.title('Cat or Not')
uploaded_file = st.file_uploader("画像をアップロードしてください", type=['jpeg', 'jpg'])

if uploaded_file is not None:
    # FastAPIに画像を送信
    files = {"file": uploaded_file.getvalue()}
    # ここでURLを文字列として正しく記述
    response = requests.post("https://catornot.onrender.com/make_predictions", files=files)

    # 結果の表示
    if response.status_code == 200:
        result = response.json()
        st.write(f'結果：{result["result"]}')
    else:
        # 'st.write' の綴りを修正
        st.write('エラー')