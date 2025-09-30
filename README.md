# Federated Multimodal Learning Project

## 프로젝트 소개

본 프로젝트는 "서로 다른 종류의 데이터(모달리티)" 를 보유한 여러 클라이언트가 자신의 데이터를 직접 공유하지 않고도 함께 AI 모델을 학습할 수 있도록 설계된 연합 학습(Federated Learning) 시스템입니다.



> MRI 영상만 있는 병원, 진료 기록만 있는 병원, 둘 다 가진 병원이 있다고 가정해 보겠습니다.
이 병원들이 자신의 데이터는 외부에 유출하지 않으면서도, 똑똑한 AI 진단 모델을 함께 학습할 수 있는 방법,
그것이 바로 본 프로젝트의 핵심입니다.

현대 의료 환경에서는 X-ray 이미지와 진단 보고서(텍스트)처럼 한 환자에 대해 여러 형태의 데이터(멀티모달)가 수집됩니다. 하지만 이 중 일부 데이터가 누락되는 '결손 양상(Missing Modality)' 문제가 자주 발생하며, 이는 의료 AI 모델의 정확성과 신뢰도를 떨어뜨리는 주된 원인이 됩니다.

### 데이터의 불완전성
- **핵심 문제**: 특정 환자의 의료 데이터 중 이미지나 텍스트 보고서 중 하나가 없는 경우, AI가 정확한 예측을 하기 어렵습니다.

- **결과**: 각 병원이 보유한 데이터의 특성에 따라 AI 성능이 달라집니다. 특히, 특정 병원에서 드물게 관찰되는 질환에 대해서는 AI가 현저히 낮은 성능을 보일 수 있습니다.

### 해결 전략: 연합학습(FL)과 FD 기술의 결합
이 문제를 해결하기 위해 **연합학습(Federated Learning, FL)**과 FD(Fusion for Missing Modality/Failure-robust Design) 기술을 결합하는 방안이 주목받고 있습니다.

- **연합학습 (FL)**: 여러 병원이 민감한 환자 데이터를 직접 공유하지 않고, 각자 학습시킨 AI 모델의 일부 정보(파라미터)만을 공유해 협력하는 기술입니다. 이를 통해 개인정보를 보호하면서 더 똑똑한 AI를 만들 수 있습니다.

- **FD 기술((Fusion for Missing Modality/Failure-robust Design)**: 이미지와 텍스트 등 서로 다른 종류의 데이터를 융합하고, 특정 데이터가 없더라도 성능 저하를 최소화하도록 보완하는 기술입니다.

결론적으로, 연합학습과 FD 기술을 함께 사용하면, 병원마다 데이터의 종류와 양이 다르거나 데이터가 불완전한 상황에서도 안정적이고 신뢰도 높은 의료 AI를 개발할 수 있는 효과적인 해결책이 됩니다.

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

### 지식 증류
클라이언트를 threshold 기반으로 나누어 그룹핑 후 성능이 더 뛰어난 선생 모델이 학생 모델에게 logit 을 전달합니다.

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

- Image+Text (26 clients)

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

`classification_01.jpynb` 
| 파일 / 함수명                      | 설명                                              |
| ----------------------------- | ----------------------------------------------- |
| `run_client_training()`       | 각 클라이언트에서 로컬 학습 수행                              |
| `extract_representations()`   | MobileNetV3 / BERT Mini로 이미지/텍스트 벡터 추출          |
| `run_clustering_and_fusion()` | K-Means, Hungarian 알고리즘 및 Cross-Attention 융합 처리 |
| `FusionClassifier`            | 최종 통합 표현 R에 대한 분류기                              |
| `global_update()`             | 클라이언트별 가중치 반영 후 글로벌 모델 업데이트                     |


## 기대 효과

- **모달리티 결손 상황에서도 성능 유지**

이미지 또는 텍스트가 누락된 상태에서도 안정적으로 진단 예측을 수행할 수 있어, 실제 임상 환경에서의 활용 가능성을 높인다.


- **연합학습 환경에서의 실질적 적용성 확보**

데이터가 병원이나 기관별로 분산되어 있는 상황에서도 프라이버시를 보장하며 학습할 수 있어, 의료 데이터 활용 범위를 확대할 수 있다.
연합학습과 지식 증류를 통해 병원별 데이터 특성을 상호 보완할 수 있어, 특정 질환에 편중된 학습 데이터의 한계를 극복하고 다양한 질환에 대한 일반화 성능을 확보할 수 있다.
각 병원이 보유하지 않은 질환 유형에 대해서도 다른 병원의 학습 경험을 공유함으로써, 보고 사례가 적은 희귀 질환이나 드문 질환에 대한 진단 예측 정확도를 높일 수 있다.


-**효율적인 지식 공유**

클러스터링 및 Representation-level KD를 통해, 모든 모달리티를 가진 클라이언트뿐만 아니라 결손된 클라이언트도 글로벌 모델 성능 향상에 기여할 수 있다.


-**일반화 성능 향상**

Cross-Attention 기반 글로벌 표현 학습을 통해, 다양한 모달리티 조합에서 강건하게 작동하는 모델을 구축할 수 있다.
결손 데이터 환경에서도 일관된 성능을 보장함으로써 의료진이 AI를 보조적 도구로 신뢰할 수 있는 기반을 마련한다. 더 나아가, 다양한 규모와 특성을 가진 병원들로 확장 적용이 가능해 의료 AI의 보급과 실질적 임상 적용을 촉진할 수 있다.


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
