import torch
import clip
from PIL import Image
from torchvision import transforms
import pandas as pd
import numpy as np
from datasets import load_dataset

# 1. CLIP 모델 로드
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 2. 기존 전처리 코드 적용 (이 부분에서 기존의 전처리 코드를 활용)
def custom_preprocessing(df):
    # 기존의 전처리 코드가 들어가는 부분
    # 예시: df['TEXT'] = df['TEXT'].apply(lambda x: x.lower().strip())
    return df

# 3. 데이터 로드 및 전처리 적용
dataset = load_dataset("laion/laion400m", split="train[:1000]")  # 1000개 샘플만 사용
df = pd.DataFrame(dataset)
df = custom_preprocessing(df)  # 기존 전처리 적용

# 4. 이미지 전처리 및 벡터화
def process_image(image_url):
    try:
        image = Image.open(image_url).convert("RGB")
        image = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image)
        return image_features.cpu().numpy()
    except Exception as e:
        print(f"Error processing image {image_url}: {e}")
        return None

# 5. 텍스트 전처리 및 벡터화
def process_text(text):
    text = clip.tokenize([text]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)
    return text_features.cpu().numpy()

# 6. 데이터셋 벡터화 수행
image_vectors = []
text_vectors = []
for i, row in df.iterrows():
    img_vector = process_image(row["URL"])
    text_vector = process_text(row["TEXT"])
    if img_vector is not None and text_vector is not None:
        image_vectors.append(img_vector)
        text_vectors.append(text_vector)

# 7. 벡터 저장 및 모델 입력으로 사용
image_vectors = np.array(image_vectors)
text_vectors = np.array(text_vectors)
np.save("image_vectors.npy", image_vectors)
np.save("text_vectors.npy", text_vectors)

# 8. 모델 입력 예제
sample_input = torch.tensor(image_vectors[:5]).to(device)  # 샘플 5개
with torch.no_grad():
    output = model.encode_image(sample_input)
print("샘플 예측 결과:", output.shape)

print("벡터 저장 완료! image_vectors.npy, text_vectors.npy")
