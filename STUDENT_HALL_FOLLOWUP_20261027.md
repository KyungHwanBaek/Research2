# 학생회관 후속 복원 실험 기록 · 2026-10-27

## 결과 요약
이전 선택 모델 PSNR 24.1397dB, SSIM 0.7736 → 이번 선택 24.6562dB, 0.7946. 선택: freeze_principal_960. Gaussian 수 436,661 → 211,669. 같은 원본 검증 사진 [20,26,32,38,44]에 대해 저장된 PNG를 동일 720×540 해상도와 함수로 비교했다.
표면 국소 보정 후보는 이전과 같은 보간 깊이 지표에서 20.3786→20.2174cm, 보간하지 않은 실제 LiDAR ray 지표에서 24.0718→23.7218cm였다. 서로 다른 두 지표의 절대 숫자는 직접 비교하지 않는다. 차이가 작으므로 실측 정확도의 뚜렷한 향상으로 주장하지 않고 기존 구조를 기본으로 유지한다. 보정 후보도 별도로 공개한다.

## 1. 자료와 기존 결과 보존
기존 01-26-15-41-24의 RGB 좌우 96장, 원시 LiDAR 스캔 48개와 보정값을 사용했다. 추가 촬영이나 친구의 누적 LiDAR 지도는 없다. 이번에 NIR를 새로 학습한 것은 아니다. 전날의 결과와 원본을 수정하지 않고 experiments/student_hall_refinement_20261027에 저장했다. 시작 시 기존 모델·표면·위치·평가 파일 5개를 SHA-256으로 기록하고 종료 시 동일성을 확인한다. 원본 파일 195개의 이전 해시도 재검사한다.

## 2. 52장에서 96장으로 COLMAP 확장
이전 SQLite DB를 SQLite backup API로 복제하여 기존 image/frame/feature ID를 보존했다. 같은 원본 해상도 1440×1080, 왜곡 보정, SIFT 최대 11,000개와 고정 내부 파라미터 조건으로 사진 44장을 추가했다. 기존 52장의 특징·대응은 재사용하고 총 4,560개 영상 쌍을 exhaustive/guided matching으로 처리했다. 기존 52장 모델에서 증분 복원을 계속하고 전역 bundle adjustment를 수행했다.
결과: 96장 중 88장(44개 촬영 시점)을 한 모델로 연결, 56,257점과 278,436개 관측, 평균 재투영 오차 0.8065px. 18·19·46·47 시점의 좌우 8장은 연결하지 못했다. 56,257점은 앞뒤 경로까지 포함한 값이므로 학생회관 같은 범위의 정확도나 밀도가 그 비율로 증가했다고 해석하지 않는다. 추가 추출 배치로 카메라 ID가 4개가 되었으나 실제 장치는 좌우 2개이며 같은 눈의 내부값은 같은 고정 보정값이다.
학생회관 관측과 100개 이상 3D 특징을 공유하는 추가 사진만 외관 학습에 넣었다. 추가 시점은 [0,1,2,3,9,10,11]의 좌우 14장이다. 학습은 42→56장, 검증은 같은 왼쪽 5장. 검증 시점의 오른쪽 사진도 학습에서 제외했다. 이 범위 확장이 항상 화질을 높이는지 별도로 평가했다.

## 3. 카메라 보정 재최적화와 동일 조건 대조군
고정 카메라 모델의 평균 재투영 오차는 약 0.8758px였다. 모든 52장이 등록된 상태에서 공유 focal만 조정한 BA는 0.6911px, focal과 principal point를 함께 조정한 BA는 0.6471px였다. 관측점이 바뀐 것이 아니라 카메라 내부값·자세·3D점을 함께 맞춘 결과다. 재투영 오차 감소를 건물 절대 치수 정확도 향상으로 해석하지 않는다. 정확한 카메라 값은 각 summary.json에 저장했다.
COLMAP 공식 FAQ의 공유 내부 보정과 전역 BA 후 principal-point 보정 지침을 확인했다. OpenCV와 COLMAP의 픽셀 중심 관례에 0.5px 차이가 있다는 점도 검토했다. 원본 이미지는 이전 방식으로 이미 왜곡 보정되어 있으며, 이번 후보는 PINHOLE 내부값을 재조정했다. 큰 잔여 왜곡까지 새로 교정했다고 주장하지 않는다.
이번 fixed/focal/principal 실험은 같은 42개 학습 사진, 검증 5개, SIFT track, seed 선택 규칙, 12,000회 예산과 seed42를 사용했다. 새 대조군부터는 초기 3D점 색상을 학습 사진에서만 다시 계산하고 학습 관측 2개 이상인 점만 남겼다. 세 모델은 동일한 28,892개 초기점을 썼다. 따라서 이전 모델의 32,820개 초기점·전체 SfM 사진 색상 사용과는 조건이 일부 다르며, 이번 내부 대조군과 이전 선택 모델의 비교를 구분한다. 전체 사진을 이용한 SfM 위치 추정은 계속 사용하므로 검증이 완전히 독립적인 것은 아니다.

## 4. 좋은 초기 모델을 보존한 고해상도 추가 학습
일반 학습에서는 Gaussian 수 증가와 약 3,000회 주기의 opacity reset 이후 화질이 낮아지는 구간이 관찰됐다. 매 1,000회마다 검증하여 가장 좋은 중간 모델을 저장했다. 이 현상이 보정 변경의 단독 원인이라고 단정하지는 않는다.
가장 좋은 중간 모델에서 Gaussian 수와 위상을 고정하고 960×720으로 추가 6,000회를 시험했다. split/clone/prune/opacity reset은 멈추지만 위치·크기·회전·색상·불투명도는 계속 학습한다. 학습 사진과 검증 분할은 같다. 960px 렌더링은 area downsample로 720×540으로 맞췄고, 마지막 모델 대신 가장 좋았던 checkpoint를 채택했다. 0.05dB 미만 변화만으로 기존 결과를 교체하지 않으며, SSIM이 0.002 이상 나빠지는 후보도 제외한다.
선택 모델의 사진별 결과는 results.json에 있다. PSNR/SSIM은 외관 일치도이고 LiDAR 구조의 미터 단위 정확도가 아니다. 실제로 학습한 최종 외관 모델의 손실은 RGB/SSIM이며 이번 모델에 LiDAR 깊이 손실이 들어갔다고 주장하지 않는다. 이전에 시도한 depth-loss 2DGS 실험은 10월26일 기록에 남아 있다.

## 5. 360도 LiDAR TSDF
이전 TSDF가 RGB 카메라 정면에 투영한 깊이만 사용했던 한계를 보완했다. 각 스캔의 물리 LiDAR 원점에서 100° 시야·512px 가상 카메라 6개를 만들고, 같은 반환점을 방향별로 투영했다. 가까운 표면 사이만 reciprocal-depth 삼각 보간했다. 이 가상 6면은 같은 스캔을 나타내므로 6개의 독립 센서 관측으로 세지 않는다.
6·10·15cm voxel, truncation 4×voxel 조건으로 검증 표면을 만들었다. 모든 후보에서 스캔 [5,13,21,29,37,45]는 통째로 제외했다. 빈 영역 벌점을 포함한 학생회관 지표가 기존보다 크게 나빠 최종에는 채택하지 않았다. 구멍이 많이 남는 것이 확인됐으며 정확한 원인이 하나라고 단정하지 않는다. 더 많은 삼각형만으로 좋은 복원이라고 판단하지 않았다.

## 6. 실제 LiDAR ray로 추가 검증
기존 지표는 960×720의 보간 깊이를 4px 간격으로 비교하고, 오차는 1m로 잘라 평균하며 예측 표면이 없는 ray도 1m 벌점을 준다. 학생회관 점수는 frame21·29·37·45 평균이다.
새 지표는 보간하지 않은 LiDAR 반환점을 네 개마다 하나씩 뽑아, 물리 LiDAR 원점에서 실제 방향과 거리로 ray casting했다. 3m 이상·35m 미만, 센서 자기 반환점 제외, 동일 1m 상한과 결측 벌점. 360° 전체와 원래 카메라 FOV 부분을 따로 기록했다. 기준 표면도 정확히 같은 ray로 다시 평가했다. 이 기준은 물리 원점·표본·거리 정의가 다르므로 기존 보간 깊이 지표 20.38cm와 새 raw 지표 24.07cm를 서로 직접 비교해서는 안 된다.
검증 스캔의 깊이는 표면 생성에서 제외했지만 카메라/센서 위치 추정에는 전체 스캔이 사용됐다. 따라서 이 역시 센서 일관성 진단이며 외부 측량 정답 시험은 아니다. 반복 선택에 사용한 validation으로, 독립 최종 test라고 부르지 않는다.

## 7. LiDAR에 맞춘 국소 평면 보정 후보
기존 Poisson 표면을 시작점으로 사용했다. 학습 스캔의 짝수 반환점만 모아 8cm voxel로 줄이고, 각 정점 주변 최대20개 점에 가중 PCA 평면을 맞췄다. 8개 이상 지지점, 작은 평면성 오차, 기존 표면 normal과 호환, 최근접 관측 15cm 이내인 부분만 움직였다. 한 번만 투영하며 새 면이나 구멍 채움을 하지 않는다.
반경25cm·이동5cm 상한 후보와 반경35cm·이동8cm 상한 후보를 비교했다. 공개하는 안전 후보는 검증 표면 정점 467,171개를 보정했고, 이동 중앙값은 6.47mm였다. 전체48스캔 최종 후보도 같은 절차이며 약212만 삼각형의 위상을 유지한다. 정확한 이동/복귀 수는 각 JSON에 보존했다. 아주 작은 면 일부에서 방향 반전이 생기는 것을 검사해, 해당 면의 정점을 원래 위치로 돌리는 단조 반복으로 제거한 안전 후보를 별도로 만들었다. 공개하는 국소 보정 후보는 이 검사를 통과한 버전이며 원래 후보도 보관했다. 원본 사진의 가시성 기반 색상도 다시 적용했다. 웹 표시는 기존과 같은 계단·입구 ROI와 정규화를 쓴다. 표시용 축소 mesh는 각각 독립적으로 줄인 것이므로 면 인덱스까지 동일한 비교는 아니다.
두 지표가 소폭 좋아졌지만 n=4 학생회관 시점이고 개선 폭이 작다. 통계적 유의성이나 실제 구조 정밀도 향상을 확정하지 않는다. 그래서 기본 구조는 이전 모델이고, 국소 보정은 비교·다운로드용 후보로 제공한다.

## 8. 재현 자료와 한계
runs/*.json에는 정확한 명령, 시각, cwd, 스레드 설정, 종료 코드와 경과 시간을, logs/*.log에는 stdout/stderr 전체를 저장했다. events.jsonl은 시간순이다. 소스와 환경, 원본/모델 hash, 체크포인트·학습 이력·검증값을 보존했다. 경과 시간은 wall clock이며 기기 대기나 스케줄링을 포함할 수 있어 GPU 계산 성능 벤치마크로 쓰지 않는다. 일부 독립 학습은 같은 GPU를 공유해 진행했으며 데이터/출력 경로는 분리했다.
검증 사진을 초기 색상에서 제외하는 개선을 했지만 SfM은 전체 사진을 이용한다. 후보를 여러 번 선택한 검증 자료에 대한 과적합 가능성도 남는다. 일부 native 렌더러 실행에서 타일 내 Gaussian 처리 상한 경고가 발생하며 로그에 보존했다. 평가값은 이 렌더러가 실제 생성한 결과의 값이다.
공개하는 간편 .splat은 시점별 SH 색상을 단순화한 변환본이며 전체 PLY/체크포인트와 표현력이 동일하지 않다. 영상과 평가 PNG는 원본 학습 모델에서 생성했다. 공개 ZIP은 명령·로그·코드·설정·수치·입출력 해시·비교 그림을 담는다. 용량이 큰 원본 사진/HDF5·전체 표면·학습 체크포인트는 중복 포함하지 않고 실험 폴더에 보존했다. 공개 사본에서는 개인 홈 경로를 /PROJECT 또는 /USER_HOME으로 치환한다. 배포 후 HTTP 확인과 배포 ID는 별도 로컬 publication/HTTP 기록에 보관한다.

## 9. 실제로 수행한 실행 목록
- full_sfm_prepare: succeeded, 3.07s, exit 0; .venv/bin/python scripts/hall_followup_sfm.py prepare
- ba_focal: succeeded, 2.54s, exit 0; .venv/bin/python scripts/hall_followup_sfm.py ba_focal
- full_sfm_features: succeeded, 17.63s, exit 0; .venv/bin/python scripts/hall_followup_sfm.py features
- cube_depth_prepare: succeeded, 21.55s, exit 0; .venv/bin/python scripts/hall_followup_surface.py prepare
- baseline_raw_lidar_rays: succeeded, 3.56s, exit 0; .venv-mesh312/bin/python scripts/hall_followup_surface.py baseline_raw
- cube_tsdf_006: succeeded, 60.95s, exit 0; .venv-mesh312/bin/python scripts/hall_followup_surface.py validation --voxel .06
- full_sfm_match: succeeded, 646.11s, exit 0; .venv/bin/python scripts/hall_followup_sfm.py match
- ba_principal: succeeded, 2.33s, exit 0; .venv/bin/python scripts/hall_followup_sfm.py ba_principal
- cube_tsdf_010: succeeded, 15.53s, exit 0; .venv-mesh312/bin/python scripts/hall_followup_surface.py validation --voxel .10
- cube_tsdf_015: succeeded, 10.30s, exit 0; .venv-mesh312/bin/python scripts/hall_followup_surface.py validation --voxel .15
- gs_fixed_prepare: succeeded, 5.12s, exit 0; .venv/bin/python scripts/hall_followup_gs.py prepare --method fixed
- gs_fixed_train: succeeded, 203.64s, exit 0; .venv/bin/python scripts/hall_followup_gs.py train --method fixed --steps 12000
- mls_025_005: succeeded, 13.43s, exit 0; .venv-mesh312/bin/python scripts/hall_followup_mls.py --radius .25 --limit .05
- gs_focal_prepare: succeeded, 5.19s, exit 0; .venv/bin/python scripts/hall_followup_gs.py prepare --method focal
- gs_principal_prepare: succeeded, 4.82s, exit 0; .venv/bin/python scripts/hall_followup_gs.py prepare --method principal
- mls_035_008: succeeded, 15.02s, exit 0; .venv-mesh312/bin/python scripts/hall_followup_mls.py --radius .35 --limit .08
- gs_focal_train: succeeded, 179.86s, exit 0; .venv/bin/python scripts/hall_followup_gs.py train --method focal --steps 12000
- full_sfm_mapping: succeeded, 41.44s, exit 0; .venv/bin/python scripts/hall_followup_sfm.py mapping
- mls_final_candidate: succeeded, 11.80s, exit 0; .venv-mesh312/bin/python scripts/hall_followup_mls.py --radius .35 --limit .08 --final
- gs_context_prepare: succeeded, 3.57s, exit 0; .venv/bin/python scripts/hall_followup_gs.py prepare --method context
- gs_principal_train: succeeded, 1139.70s, exit 0; .venv/bin/python scripts/hall_followup_gs.py train --method principal --steps 12000
- gs_context_train: succeeded, 269.29s, exit 0; .venv/bin/python scripts/hall_followup_gs.py train --method context --steps 12000
- freeze_principal_960: succeeded, 191.74s, exit 0; .venv/bin/python scripts/hall_followup_freeze.py --method principal --width 960 --steps 6000
- mls_display_and_evidence: succeeded, 18.98s, exit 0; .venv/bin/python scripts/hall_followup_publish.py geometry_assets
- common_evaluation_and_report: succeeded, 2.69s, exit 0; .venv/bin/python scripts/hall_followup_publish.py report_and_assets
- mls_orientation_validation: succeeded, 9.02s, exit 0; .venv-mesh312/bin/python scripts/hall_followup_mls.py --radius .35 --limit .08 --safe
- mls_orientation_final: succeeded, 11.42s, exit 0; .venv-mesh312/bin/python scripts/hall_followup_mls.py --radius .35 --limit .08 --safe --final
- safe_mesh_display: succeeded, 9.98s, exit 0; .venv/bin/python scripts/hall_followup_publish.py geometry_assets
- finalized_evaluation_and_report: succeeded, 3.14s, exit 0; .venv/bin/python scripts/hall_followup_publish.py report_and_assets
- final_provenance: succeeded, 1.45s, exit 0; .venv/bin/python scripts/hall_followup_publish.py provenance

## 참고한 공식 자료
- COLMAP FAQ: https://colmap.github.io/faq.html
- Open3D RGB-D/TSDF integration: https://www.open3d.org/docs/release/tutorial/pipelines/rgbd_integration.html
국소 PCA 투영과 검증 절차는 이 프로젝트 구현이며 특정 논문의 완전 재현으로 주장하지 않는다.
