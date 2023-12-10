import streamlit as st
import requests
from io import BytesIO

st.title('Cat Or Not')
uploaded_file = st.file_uploader("画像をアップロードしてください", type=['jpeg', 'jpg'])

if uploaded_file is not None:
    # FastAPIサーバーに画像を送信
    files = {"file": uploaded_file.getvalue()}
    response = requests.post("https://catornot.onrender.com/make_predictions", files=files)
           #response = requests.post("http://127.0.0.1:8000/make_predictions",files=files)

    # 結果の表示
    if response.status_code == 200:
        result = response.json()
        st.write(f'結果：{result["result"]}')
    else:
        st.error(f'エラーが発生しました: {response.status_code}')
        st.write(response.text)


 