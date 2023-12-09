import streamlit as st
import pandas as pd
import requests

st.title('Cat or Not')
uploaded_file = st.file_uploader("画像をアップロードしてください", type='jpeg')

if uploaded_file is not None:
    # FastAPIに画像を送信
    files = {"file": uploaded_file.getvalue()}
    response = requests.post(, files=files)

    # 結果の表示
    if response.status_code == 200:
        result = response.json()
        st.write(f'結果：{result["result"]}')
    else:
        st.wite('エラー')