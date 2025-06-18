# Federated Multimodal Learning Project

## 프로젝트 소개

본 프로젝트는 "서로 다른 종류의 데이터(모달리티)" 를 보유한 여러 클라이언트가 자신의 데이터를 직접 공유하지 않고도 함께 AI 모델을 학습할 수 있도록 설계된 연합 학습(Federated Learning) 시스템입니다.



> MRI 영상만 있는 병원, 진료 기록만 있는 병원, 둘 다 가진 병원이 있다고 가정해 보겠습니다.
이 병원들이 자신의 데이터는 외부에 유출하지 않으면서도, 똑똑한 AI 진단 모델을 함께 학습할 수 있는 방법,
그것이 바로 본 프로젝트의 핵심입니다.

## 문제 상황
### 1. 서로 다른 정보만 보유한 클라이언트
어떤 클라이언트는 이미지 데이터만 보유하고 있으며,

또 어떤 클라이언트는 텍스트 데이터만 보유하고 있습니다.

두 가지를 모두 가진 클라이언트는 드뭅니다.

### 2. 데이터를 공유할 수 없는 환경
개인정보 보호와 민감 정보 유출 방지 등의 이유로 로컬 데이터를 외부에 공유하는 것이 불가능합니다.

## 해결 아이디어
각 클라이언트는 자신이 보유한 데이터만을 이용해 모델을 학습합니다.

서버는 모든 클라이언트의 모델 가중치(또는 벡터 정보)를 모아 융합합니다.

부족한 모달리티 정보는 글로벌 벡터를 통해 보완합니다.

단일 모달리티만 가진 클라이언트도, 다른 클라이언트의 도움을 간접적으로 받아 성능을 향상시킬 수 있습니다.

## 전체 구조
```
CLIENT_ROOT/
│
├── client_0/
│   ├── train/   ← 이미지 또는 텍스트 등의 학습 데이터
│   ├── val/     ← 검증용 데이터
│   ├── best_model.pt ← 학습된 로컬 모델
│
├── ...
├── server/
│   └── global_vectors/ ← 클라이언트들이 업로드한 벡터 정보
```

## 모델 구조 (Fusion Classifier)
- 이미지 인코더: MobileNet V3

- 텍스트 인코더: BERT Tiny

- 융합 방식: late fusion + gating mechanism
→ 두 벡터를 단순히 합치는 것이 아니라, 가중치를 조절하여 중요한 정보에 더 집중하도록 설계하였습니다.

## 학습 흐름
### 로컬 학습

클라이언트는 자신의 데이터만을 활용하여 모델을 학습합니다.

### 글로벌 벡터 업로드

학습된 결과 중 핵심 정보(벡터)를 서버에 전송합니다.

### 서버 측 융합

서버는 여러 클라이언트의 벡터를 모아 전역 정보를 구성합니다.

### 글로벌 정보 반영

클라이언트는 이 전역 정보를 활용하여 부족한 모달리티를 보완합니다.

### 반복 학습

위 과정을 여러 차례 반복함으로써 전체 모델 성능을 점차 향상시킵니다.

## 코드 체계
### 📁 1. Local Area (로컬 학습 영역)

각 클라이언트는 다음 중 하나의 모달리티 구성에 해당합니다:

- Image-only (2 clients)

- Text-only (2 clients)

- Image+Text (6 clients)

사용된 모델:

- MobileNetV3: 이미지 인코더

- BERT Mini: 텍스트 인코더

수행 로직:

- 공통 데이터셋(서버에서 분배됨)을 통해 각 클라이언트가 로컬 학습 수행

- image_encoder() / text_encoder()를 통해 표현 벡터 추출

- 클라이언트별로 local encoder output 저장 (e.g., img_vec.pt, txt_vec.pt)

### 📁 2. Global Area (클러스터링 및 융합 영역)
표현 벡터 수집 및 클러스터링:

- 각 클라이언트로부터 수집된 이미지/텍스트 벡터에 대해 K-Means 적용

- Hungarian Algorithm으로 cross-modality pair 최적화

**Late Fusion:**

- 각 modality 클러스터에서 중심 벡터(center vector)를 추출하여 Late Fusion

**Cross-Attention & Gating:**

- CrossAttentionTransformer: 이미지-텍스트 중심 벡터 간 핵심 정보 Z 추출

- GatingMechanism: forget gate 기반으로 최종 통합 표현 R 생성

### 📁 3. Classification & Global Update (분류 및 글로벌 업데이트 영역)
분류:

- 통합 표현 R을 입력으로 하는 FusionClassifier 수행

- Cross-entropy loss로 최종 classification 결과 도출

**글로벌 업데이트:**

- 클라이언트별 adapter layer(Fusion Classifier 앞단) 성능 기반 기여도를 계산

- 계산된 기여도를 바탕으로 가중 평균 방식으로 글로벌 모델 업데이트

- adapter weight(로컬에서 새로 계산된 weight)는 서버에서 통합 후 재배포하, local encoder는 클라이언트가 유지

`classification_01.jpynb` 파일 설명
| 파일 / 함수명                      | 설명                                              |
| ----------------------------- | ----------------------------------------------- |
| `run_client_training()`       | 각 클라이언트에서 로컬 학습 수행                              |
| `extract_representations()`   | MobileNetV3 / BERT Mini로 이미지/텍스트 벡터 추출          |
| `run_clustering_and_fusion()` | K-Means, Hungarian 알고리즘 및 Cross-Attention 융합 처리 |
| `FusionClassifier`            | 최종 통합 표현 R에 대한 분류기                              |
| `global_update()`             | 클라이언트별 가중치 반영 후 글로벌 모델 업데이트                     |


## 기대 효과
- 모달리티 결손 문제 해결: 부족한 정보를 글로벌 벡터를 통해 보완할 수 있습니다.

- 개인정보 보호 보장: 원본 데이터는 절대 외부로 유출되지 않습니다.

- 모델 성능 향상: 클라이언트들이 서로 간접적으로 도움을 주고받으며 성능이 향상됩니다.

## 실행 방법

# 예시 실행 코드 (Python 가상환경 활성화 후)
```
python train_clients.py  # 모든 클라이언트 개별 학습
python aggregate_vectors.py  # 서버에서 벡터 융합
python update_clients.py  # 클라이언트가 글로벌 정보 반영
※ 실제 실행 시, 클라이언트별 디렉토리 구성과 환경 설정이 필요합니다.
```

## 사용 기술
- Python, PyTorch

- HuggingFace Transformers

- torchvision (MobileNetV3)

- 연합 학습 구조 (custom FL pipeline)

- Gating Mechanism 기반 Late Fusion
