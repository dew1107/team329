# 개발팀 그라운드룰

## 1️⃣ 개발 목표
### 📌 프로젝트 개요
- 본 프로젝트는 **이미지와 텍스트의 크로스어텐션과 코어텐션을 활용하여 글로벌 컨텍스트를 반영한 멀티모달 표현 학습 모델**을 개발하는 것을 목표로 한다.
- 기존 모델을 바탕으로, 효율적인 정보 융합 및 멀티모달 학습을 최적화하는 방향으로 개발을 진행한다.

## 2️⃣ 코드 스타일 & 버전 관리
### 📝 코드 스타일
- **PEP8** (Python) 및 **Google Style Guide** 준수
- 변수 및 함수명: **snake_case** 사용 (`extract_features_from_image`)
- 클래스명: **PascalCase** 사용 (`MultimodalFusionModel`)
- 주석 필수 (`# 설명` 또는 `"""Docstring"""` 활용)

### 🗂️ Git 규칙
- **메인 브랜치 보호:** `main` 브랜치에 직접 푸시 금지, PR(풀 리퀘스트)로 병합
- **브랜치 네이밍 규칙:** `feature/{기능명}`, `fix/{버그명}`, `refactor/{리팩토링명}`
- **커밋 메시지 스타일:**
  - `[Feat] 새로운 기능 추가`
  - `[Fix] 버그 수정`
  - `[Refactor] 코드 리팩토링`
  - `[Docs] 문서 수정`
  - `[Test] 테스트 코드 추가`

## 3️⃣ 커뮤니케이션
### 📢 팀 소통 원칙
- **주 1회 정기 미팅 진행** (온라인/오프라인 병행)
- 미팅 전 **아젠다 공유**, 미팅 후 **요약 정리** 필수
- GitHub Issues 및 Slack을 활용하여 이슈 및 논의 사항 관리

## 4️⃣ 개발 워크플로우
### 🚀 개발 단계
1. **기능 정의:** 각 기능을 GitHub Issues에 등록
2. **브랜치 생성:** `feature/{기능명}` 브랜치에서 개발
3. **PR 요청 & 코드 리뷰:** 최소 2인의 승인 후 `main` 브랜치에 병합
4. **테스트 & 배포:** 기본 테스트 코드 통과 후 병합 진행

### 📌 역할 분담 
| 역할  | 담당자 |
|------|------|
| 모델 아키텍처 설계 | 이혜리 |
| 데이터 전처리 & 학습 파이프라인 구축 | 김희진 |
| 실험 & 성능 최적화 | 이슬 |

## 5️⃣ 코드 리뷰 & 테스트
### ✅ 코드 리뷰 원칙
- 모든 PR은 최소 **1명 이상의 승인 필요**
- 리뷰 시 **구체적인 피드백 제공** (`변경 이유 + 개선 제안` 포함)
- `LGTM`(Looks Good To Me) 대신, 반드시 **간단한 개선 사항 제안**

### 🔍 테스트 정책
- 모든 모델 및 기능에 대한 **유닛 테스트 필수**
- `pytest` 및 `unittest` 활용
- 코드 커버리지 80% 이상 유지 목표
- 성능 평가 지표 (예: Accuracy, F1-score, BLEU 등) 정리 및 공유



