# モジュールのインポート
from fastapi import FastAPI, File, UploadFile
from io import BytesIO
from PIL import Image
import torch
from torchvision import models, transforms

# インスタンス化
app = FastAPI()

# 画像の前処理関数
def preprocess_image(image: Image.Image) -> torch.Tensor:
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(image)

# モデルのロード
#def load_model():
#    model = models.vgg16(pretrained=True)
#    model.eval()
#    return model

def load_model():
    # MobileNet V2モデルを読み込む
    model = models.mobilenet_v2(pretrained=True)
    model.eval()
    return model

# トップページ
@app.get('/')
async def read_root():
    return {"message": "Welcome to the Image Classifier API"}

# POST が送信された時（入力）と予測値（出力）の定義
@app.post('/make_predictions')
async def create_upload_file(file: UploadFile = File(...)):
    try:
        # 画像の読み込みと前処理
        image = Image.open(BytesIO(await file.read())).convert("RGB")
        input_tensor = preprocess_image(image)
        input_batch = input_tensor.unsqueeze(0)

        # モデルのロード
        model = load_model()

        # 推論
        with torch.no_grad():
            output = model(input_batch)

        # ImageNetのクラスIDを使用して、猫かどうかを判定
        cat_classes_ids = [281, 282, 283, 284, 285]  # シャム猫, ペルシャ猫 など
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        top_prob, top_catid = torch.topk(probabilities, 1)
        result = "猫" if top_catid.item() in cat_classes_ids else "それ以外"
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}



'''
# モジュールのインポート
from fastapi import FastAPI, File, UploadFile
from io import BytesIO
from PIL import Image
import torch
from torchvision import models, datasets, transforms

# インスタンス化
app = FastAPI()

# データセットの変換を定義
transform = transforms.Compose([
    transforms.ToTensor()
])

# データセットの取得
train_val = datasets.CIFAR10('./', train=True, download=True, transform=transform)
test = datasets.CIFAR10('./', train=False, download=True, transform=transform)

# 学習済みモデルの読み込み
model = models.vgg16(pretrained=True)
model.eval()

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
    return {"message": "Welcome to the Image Classifier API"}

# ImageNetの猫のクラスIDリストの定義（例）
# これはImageNetの特定の猫のクラスIDの一例です。実際にはもっと多くの猫のクラスがあります。
cat_classes_ids = [281, 282, 283, 284, 285]  # シャム猫, ペルシャ猫 など

# POST が送信された時（入力）と予測値（出力）の定義
@app.post('/make_predictions')
async def create_upload_file(file: UploadFile = File(...)):
    try:
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
        result = "猫" if top_catid.item() in cat_classes_ids else "それ以外"
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}
'''