import pandas as pd

# --- 설정 ---
# 원본 파일 경로
path_test_labels = 'mimic-cxr-2.1.0-test-set-labeled.csv'
path_metadata = 'mimic-cxr-2.0.0-metadata.csv'

# 새로 저장할 파일 이름
path_output_fixed = 'mimic-cxr-2.1.0-test-set-labeled-FIXED.csv'

# 스크립트(vector_test_local.py)가 요구하는 필수 열 이름
# (negbio.csv의 헤더를 기준으로 함)
REQUIRED_COLUMNS = [
    'subject_id', 'study_id',
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion',
    'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity',
    'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices'
]

# --- 1. 메타데이터에서 (study_id -> subject_id) 맵 생성 ---
print(f"'{path_metadata}' 로딩 중...")
df_meta = pd.read_csv(path_metadata)
# 중복 제거 (study_id당 subject_id는 유일함)
study_to_subject_map = df_meta[['study_id', 'subject_id']].drop_duplicates().set_index('study_id')[
    'subject_id'].to_dict()
print(f"스터디 ID {len(study_to_subject_map)}개에 대한 subject_id 맵 생성 완료.")

# --- 2. 테스트 라벨 파일 로드 및 수정 ---
print(f"'{path_test_labels}' 로딩 중...")
df_test = pd.read_csv(path_test_labels)

# 2-1. 'Lung Opacity' 열 이름 변경
if 'Airspace Opacity' in df_test.columns:
    df_test = df_test.rename(columns={'Airspace Opacity': 'Lung Opacity'})
    print("'Airspace Opacity' -> 'Lung Opacity'로 열 이름 변경 완료.")
else:
    print("[경고] 'Airspace Opacity' 열을 찾을 수 없습니다. 이미 변경되었을 수 있습니다.")

# 2-2. 'subject_id' 열 추가
if 'subject_id' not in df_test.columns:
    df_test['subject_id'] = df_test['study_id'].map(study_to_subject_map)
    print("'subject_id' 열 추가 완료.")

    # subject_id를 찾지 못한 경우 (NaN) 확인
    missing_count = df_test['subject_id'].isna().sum()
    if missing_count > 0:
        print(f"[경고] {missing_count}개 행의 subject_id를 메타데이터에서 찾지 못했습니다. (NaN으로 채워짐)")
else:
    print("[정보] 'subject_id' 열이 이미 존재합니다.")

# --- 3. 최종 파일 저장 ---
# 스크립트가 요구하는 순서대로 열 정렬 (없는 열은 무시)
final_columns = [col for col in REQUIRED_COLUMNS if col in df_test.columns]
# 필수 열 외에 'No Finding' 등 다른 열도 있다면 포함
other_columns = [col for col in df_test.columns if col not in REQUIRED_COLUMNS]

df_final = df_test[final_columns + other_columns]

df_final.to_csv(path_output_fixed, index=False)
print(f"\n[성공] 수정된 라벨 파일 저장 완료: {path_output_fixed}")
print("이 파일 경로를 --label_csv 인자로 사용하세요.")