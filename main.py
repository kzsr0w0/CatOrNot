
# モジュールのインポート
from fastapi import FastAPI, File, UploadFile
from io import BytesIO
from PIL import Image
import numpy as np
import torch
from torchvision import models, transforms

import pickle

# インスタンス化
app = FastAPI()

# 学習済みモデルの読み込み
## VGG16モデルは自分でモデルを作らなくても外部から読み込める
model = models.vgg16(pretrained=True) # イコールでつなぐのではなく、関数で返す
model.eval() # 推論モード

# 画像の前処理関数
def preprocess_image(image: Image.Image) -> torch.Tensor:
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(image)

# トップページ
@app.get('/')
async def read_root():
    return {"message": "Welcome to the Cat or Not API"}

# 猫のクラスIDリストの定義（例）
cat_classes_ids = [281, 282, 283, 284, 285]  # これらはImageNetの特定の猫のクラスID

# POST が送信された時（入力）と予測値（出力）の定義
@app.post('/make_predictions')
async def create_upload_file(file: UploadFile = File(...)):
    # 画像の読み込みと前処理
    image = Image.open(BytesIO(await file.read())).convert("RGB")
    input_tensor = preprocess_image(image)
    input_batch = input_tensor.unsqueeze(0)

    # 推論
    with torch.no_grad():
        output = model(input_batch)

    # 猫かどうかの判定
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top_prob, top_catid = torch.topk(probabilities, 1)

    # ImageNetのクラスIDを使用して、猫かどうかを判定
    if top_catid.item() in cat_classes_ids:
        result = "猫"
    else:
        result = "それ以外"
    
    return {"result": "猫" or "それ以外"}