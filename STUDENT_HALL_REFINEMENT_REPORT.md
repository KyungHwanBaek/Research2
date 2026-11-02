# 학생회관 추가 3D 복원 실험 보고서 · 2026-10-26

## 1. 최종 결과
외관은 개선되었고, 측정 구조의 정확도는 뚜렷한 개선을 확인하지 못했다. 그래서 사진 기반 외관 모델을 교체하고, 기존 LiDAR 표면의 구조를 유지한 채 사진 색상을 붙이는 방식을 개선했다.
- 같은 검증 사진 5장의 PSNR 평균: 23.2792 → 24.1397 dB.
- 같은 함수로 계산한 SSIM 평균: 0.7216 → 0.7736.
- 기존 외관 모델 초기 특징점: 3,518개. 새 좌우 사진 COLMAP: 32,820개(약 9.3배). 이 배율은 정확도 배율이 아니다.
- 사진 색상 정점: 120,362/1,110,732 (10.84%) → 397,611/1,110,732 (35.80%). 색상 적용 비율이며 새로 측정한 구조의 비율이 아니다.
- 구조 기준: 학생회관 검증 시점 4개의 평균 페널티 포함 오차 20.3786cm. 기존 표면 유지.
- 선택: gs3d_fine_960, 체크포인트 12,000회, Gaussian 436,661개. 학습 42장(21개 시점의 좌우 사진), 검증 5장(같은 시점의 오른쪽 사진도 학습 제외). 수치는 저장된 결과를 같은 720×540 해상도로 다시 계산했습니다.

## 2. 정확히 어떤 자료를 사용했나
원본 자료는 data/extracted/01-26-15-41-24이다. 48개 촬영 시점에 RGB 좌우와 NIR 좌우, 원시 LiDAR 스캔 48개, calibration.npz, cam2lidar.npy가 있다. 친구가 별도로 만든 누적 LiDAR 지도나 이동 궤적은 제공받지 않았다.
전체 스테레오 깊이 시험에는 RGB 좌우 48쌍과 NIR 좌우 48쌍을 사용했다. 새 COLMAP과 외관 학습은 학생회관 구간 프레임 20~45의 RGB 좌우 총 52장에 집중했다. 영상 크기는 원본 1440×1080이며, 원본 왜곡 보정 후 특징점을 계산했다.
기존 학생회관 COLMAP은 두 구성요소 합계 5,183점이었고, 기존 외관 모델은 그중 주 구성요소 3,518점으로 시작했다. 이번 32,820점은 더 많은 카메라 관측과 높은 특징 추출 해상도를 함께 적용한 결과이므로 한 설정만 바꾼 실험으로 해석하지 않는다.

## 3. 보정과 스테레오 깊이 시험
카메라 기준 길이 0.132133m를 유지했다. 특징 대응으로 보정을 점검했으며, 제약 없는 essential 보정은 이동 방향이 반전되어 거부했다. LiDAR 깊이가 있는 특징의 오른쪽 영상 재투영을 추가하고 물리적으로 작은 보정으로 제한했다. 보정에 쓰지 않은 좌우 쌍에서 수직 오차 중앙값은 약 0.94→0.55px로 감소했다. 이것만으로 깊이 정확도가 검증되는 것은 아니다.
SGBM, 좌우 일관성, 질감·시차·거리 필터로 RGB/NIR 깊이를 생성했다. 프레임에 따라 특히 먼 벽에서 오차가 컸다. 각 프레임의 희소 LiDAR anchor를 80/20 공간 분할하여 시차 scale/bias 보정을 검증했다. 원본 예측은 raw_rgb_depth/raw_nir_depth로 보존했다.
RGB 깊이가 인접 다른 시점 두 개 이상과 맞고, 자유 공간을 침범하지 않으며, 같은 시점 LiDAR와도 모순되지 않는 경우만 사용했다. LiDAR 깊이가 우선하며 stereo는 빈 부분만 보충했다. NIR 깊이는 별도 비교만 했고 정렬이 검증되지 않은 독립 보완 깊이로 합치지 않았다.
검증 표면을 만들 때 프레임 [5,13,21,29,37,45]의 LiDAR/사진 깊이를 모두 제외했고, 다른 프레임의 다중 시점 필터에서도 이 시점들을 참조하지 않았다. stereo 보완은 후보 표면을 만들었지만 기준 표면보다 학생회관 검증 오차가 커 최종에는 넣지 않았다.

## 4. 구조 개선 후보와 평가
Open3D 0.19에서 원시 LiDAR를 3m 이내 센서 자기 반환점 제외 후 처리했다. 더 작은 voxel과 더 깊은 Poisson을 시험했다. 관측 점에서 너무 먼 추정 표면 및 작은 조각을 제거했다. Poisson 경고도 로그에 남겼으며 출력의 유한 좌표와 표면을 별도 확인했다.
기존 표면 생성 함수를 그대로 재실행한 대조군은 기존 검증값을 정확히 재현했다. baseline_parameters 이름의 두 후보는 원래 함수의 per-scan 12cm / merged 10cm voxel, normal radius 40cm를 사용한다. 명령줄 기본 --voxel 6cm는 이 함수에서 사용되지 않는다.
전체 스캔 보정은 이미 수락된 시점 쌍의 대칭 point-to-plane 대응을 반복 갱신하여 약 74,700개 제약을 함께 최적화했다. 짝수 반환점으로 맞추고 홀수 반환점으로 비교했다. 같은 37개 시점 쌍의 clipped plane mean은 8.1439→8.1305cm로 아주 조금 줄었지만, 학생회관 표면 검증은 20.38→20.52cm로 좋아지지 않았다. 최종 구조에는 채택하지 않았다.
표면 검증은 여섯 스캔을 표면 생성에서 통째로 빼고 ray casting으로 예측 깊이를 비교한다. 기존과 같은 LiDAR 보간 깊이, 4px 간격, 1m 오차 상한을 사용하고 표면이 없는 ray도 1m로 벌점 처리했다. 카메라 위치 추정에는 전체 스캔이 쓰였으므로 독립 실측 정답 시험이 아니다. 학생회관 점수는 프레임 21·29·37·45별 점수 평균이다.

## 5. 새로운 외관 복원
학생회관 26개 시점의 왼쪽·오른쪽 RGB 총 52장을 원래 해상도에서 왜곡 보정했다. COLMAP CPU SIFT 최대 11,000개, exhaustive matching, guided matching, 카메라별 고정 내부 파라미터로 증분 SfM과 bundle adjustment를 수행했다. 52장 모두 한 구성요소에 연결되었고 32,820개 특징점, 177,306개 관측, 원본 해상도 평균 재투영 오차 약 0.876px를 얻었다.
프레임 20·26·32·38·44는 외관 검증용이다. 이 다섯 시점의 좌우 사진을 모두 학습에서 제외하여 42장(21시점×좌우)으로 Metal 3DGS를 학습했다. 기존 비교 모델은 같은 21개 시점의 왼쪽 사진만 사용했다. 이 단계의 학습 손실은 RGB/SSIM이며 LiDAR 깊이 손실을 넣은 최종 모델로 주장하지 않는다.
16,000회 중 2,000회마다 검증했다. 마지막 모델보다 앞선 체크포인트가 좋아 이를 보관했다. Gaussian 수를 고정한 720·960px 추가 학습도 각각 6,000회 비교했다. 960px 결과는 동일 720×540 검증 해상도로 area downsample했다. 전체 후보를 같은 PNG 기준 PSNR과 Gaussian SSIM 함수로 다시 계산했고, 0.02dB 이내는 더 단순한 후보를 유지했다. 새 모델의 개선은 관측 수·해상도·SfM 모델·학습 초기화가 함께 달라진 전체 처리 과정의 개선이다.
Metal 2DGS의 실제 깊이 미분도 시험했다. 동일 LiDAR surfel 초기화에서 RGB-only와 depth-loss를 각각 4,000회 학습했다. RGB-only 약 16.10dB, depth-loss 약 15.13dB로 기존 외관보다 낮았다. 기존 학습 3DGS를 얇은 2D surfel로 바꾸는 초기화도 시험했지만, 화질 최고 약 23.02dB이고 예상 깊이에 큰 잔차가 남았다. 결과를 보관하고 채택하지 않았다. 2DGS 깊이는 alpha-weighted 합을 alpha로 나누며, 해당 렌더러의 투과율은 forward-only이다. 따라서 완전한 alpha 미분까지 구현한 깊이 최적화라고 주장하지 않는다. 깊이 중앙값은 표면이 렌더링된 픽셀에 조건부인 진단값이며 전체 구조 정확도로 쓰지 않았다.

## 6. LiDAR와 새 COLMAP을 직접 합치는 시험
새 SfM 특징의 왼쪽 영상 위치에 대응하는 LiDAR 깊이와 기존 48개 LiDAR 카메라 위치로 3D anchor를 만들었다. 유효 영역과 깊이 불연속을 거른 뒤 robust similarity를 맞췄고, landmark 20%는 이 정렬에서 제외했다. 이 anchor의 위치 차이 중앙값은 22.73cm, 90백분위 69.12cm이다.
3개 이상 다른 시점에서 관측되고 기존 컬러 LiDAR에서 20cm 이내인 새 특징 3,502개만 더한 점군도 만들었다. 그러나 이 결과는 정밀 결합으로 검증되지 않아 최종 구조에 사용하지 않았다. 전역 similarity만으로는 SfM과 LiDAR의 시점별 작은 변형을 모두 없앨 수 없다. expanded_colmap_metric / supported_fusion 파일은 후보이다.

## 7. 표면 색상을 개선한 방법
기존에는 각 표면 정점에 가장 가까운 컬러 LiDAR 점의 색을 옮겨, 관측 가능한 벽도 회색으로 남는 경우가 많았다. 이번에는 기존 표면의 모든 정점을 각 사진에 투영하고 ray casting으로 다른 표면에 가려지는지 검사했다. 시선 각도, 거리와 영상 가장자리 위치에 따라 가중치를 주고 비슷하게 좋은 여러 관측의 실제 RGB를 조합했다. 사진에 보이지 않은 정점은 회색으로 남겼다.
이것은 관측 사진을 표면에 입힌 것이다. 새 정밀 치수를 측정하거나 빈 건물 면을 새로 만들어 낸 것이 아니다. 전체 표면의 정점 좌표와 삼각형이 기존과 완전히 같음을 배열 단위로 확인했다. 웹 전후 비교도 같은 18만 삼각형의 좌표·면 버퍼를 공유하고 색상만 다르다. 전체 모델은 약 212만 삼각형으로 로컬에 보관했다.

## 8. 기록, 재현, 실패
실행별 runs/*.json: 정확한 인자, 작업 경로, 시각, 종료 코드, 경과 시간, 스레드 설정. logs/*.log: 전체 stdout/stderr. events.jsonl: 순차 실행 기록. 코드 사본은 source_snapshot/, 원본 195개 파일 해시는 raw_input_manifest.json, 환경은 environment.json과 dependencies_*.txt이다.
기존 주요 파일 9개의 시작/종료 SHA-256이 같음을 확인했다. 실패한 OpenCV 번역벡터 형상 오류는 3×1로 고쳤고, torch와 pycolmap의 OpenMP 중복 초기화 충돌은 같은 프로세스에서 pycolmap을 불러오지 않고 텍스트 카메라를 읽어 해결했다. 위험한 중복 OpenMP 허용 옵션은 쓰지 않았다. 제약 없는 카메라 보정과 성능이 낮은 후보도 삭제하지 않았다.
브라우저 자동화는 사용 가능한 브라우저가 없어 실제 UI 조작 검사를 못 했다. 대신 모델 버퍼·좌표 동일성·유한 값·정점 색상·생성 이미지·영상 정보·HTML/JS·배포 후 HTTP 응답을 확인했다. 일부 neural 렌더링에서 per-tile Gaussian 처리 상한 경고가 발생했으며 로그에 보존했다. 이 렌더러로 실제 나온 결과를 평가했고 모든 타일의 기여가 정확히 보존되었다고 주장하지 않는다.
배포 기록과 추가 검사는 로컬 QA/배포 메타데이터에 보관된다. 공개 기록 ZIP은 실험 명령·로그·설정·수치·코드·입출력 해시를 담는다. 용량이 큰 원본 사진·HDF5·전체 메시·학습 체크포인트는 ZIP에 중복하지 않고 프로젝트의 원래 경로와 실험 폴더에 보관했다. 공개 ZIP에서 개인 홈 경로는 /PROJECT 또는 /USER_HOME으로 치환한다.

## 9. 다음에 구조 정확도를 더 올리려면
이번 결과는 기존 자료에서 외관을 개선한 결과다. 더 정밀한 한 모델을 위해서는 좌우 카메라와 LiDAR를 동시에 제약하는 bundle adjustment, 센서 시간 차이와 이동 중 LiDAR 왜곡 보정, 학생회관 난간·유리·가려진 면의 추가 관측을 우선 검토해야 한다. 제공되지 않은 친구의 지도나 궤적이 이미 있다고 가정하지 않았다.

## 실행 목록
- 2dgs_depth_4000: succeeded, 368.80s, 종료 0; .venv-gs-refine/bin/python scripts/refine_student_hall_2dgs.py --method depth --steps 4000
- 2dgs_rgb_control_4000: succeeded, 403.47s, 종료 0; .venv-gs-refine/bin/python scripts/refine_student_hall_2dgs.py --method rgb --steps 4000
- 2dgs_warm_depth_2000: failed, 4.18s, 종료 -6; .venv-gs-refine/bin/python scripts/refine_student_hall_2dgs.py --warmstart --method depth --steps 2000 --eval-every 500
- 2dgs_warm_depth_2000_textposes: succeeded, 247.89s, 종료 0; .venv-gs-refine/bin/python scripts/refine_student_hall_2dgs.py --warmstart --method depth --steps 2000 --eval-every 500
- baseline_surface_control: succeeded, 35.21s, 종료 0; .venv-mesh312/bin/python scripts/refine_student_hall_surface.py --baseline-build
- fetch_metalsplat: succeeded, 2.67s, 종료 0; git clone --depth 1 https://github.com/tchauffi/metalsplat.git third_party/metalsplat
- fetch_msplat_source: succeeded, 0.81s, 종료 0; git clone --depth 1 https://github.com/rayanht/msplat.git third_party/msplat
- final_artifact_checks: succeeded, 1.69s, 종료 0; .venv/bin/python scripts/verify_student_hall_refinement.py
- finalize_source_provenance: succeeded, 0.64s, 종료 0; .venv/bin/python scripts/record_refinement_provenance.py
- fine_720: succeeded, 107.39s, 종료 0; .venv/bin/python scripts/refine_student_hall_finetune.py --width 720 --steps 6000
- fine_960: succeeded, 137.97s, 종료 0; .venv/bin/python scripts/refine_student_hall_finetune.py --width 960 --steps 6000
- fine_poisson_06_11: succeeded, 29.57s, 종료 0; .venv-mesh312/bin/python scripts/refine_student_hall_surface.py --voxel .06 --depth 11
- fine_poisson_08_12: succeeded, 50.50s, 종료 0; .venv-mesh312/bin/python scripts/refine_student_hall_surface.py --voxel .08 --depth 12
- global_multiscan_refinement: succeeded, 28.73s, 종료 0; .venv/bin/python scripts/refine_student_hall_poses.py
- global_surface_validation: succeeded, 35.13s, 종료 0; .venv-mesh312/bin/python scripts/refine_student_hall_surface.py --poses global --baseline-build
- gpu_dependencies: succeeded, 23.06s, 종료 0; .venv-gs-refine/bin/python -m pip install -e ./third_party/metalsplat pillow plyfile opencv-python scikit-image
- gpu_env_create: succeeded, 1.42s, 종료 0; .venv-mesh312/bin/python -m venv .venv-gs-refine
- hybrid_poisson_08_11: succeeded, 29.46s, 종료 0; .venv-mesh312/bin/python scripts/refine_student_hall_surface.py --mode hybrid --voxel .08 --depth 11
- metalsplat_smoke: succeeded, 11.80s, 종료 0; .venv-gs-refine/bin/python third_party/metalsplat/examples/fit_image_2dgs.py
- pose_comparison_same_normals: succeeded, 4.05s, 종료 0; .venv/bin/python -c import sys,json,numpy as np; from pathlib import Path; sys.path.insert(0,"scripts"); import refine_student_hall_poses as m; p=m.OUT/"global_pose_evaluation.json"; old=json.loads(p.read_text()); raw=[np.load(m.B/"frames"/f"{i:06d}.npz")["camera"] for i in range(48)]; r=m.evaluate(np.load(m.OUT/"global_poses.npz")["poses"],raw); r["fit_history"]=old["fit_history"]; p.write_text(json.dumps(r,indent=2)); print(json.dumps(r))
- prepare_public_results: succeeded, 3.59s, 종료 0; .venv/bin/python scripts/prepare_refinement_site.py
- record_provenance: succeeded, 1.27s, 종료 0; .venv/bin/python scripts/record_refinement_provenance.py
- refresh_provenance: succeeded, 0.88s, 종료 0; .venv/bin/python scripts/record_refinement_provenance.py
- render_comparison_evidence: succeeded, 4.86s, 종료 0; .venv-mesh312/bin/python scripts/render_refinement_evidence.py
- stereo_bias: succeeded, 8.09s, 종료 0; .venv/bin/python scripts/refine_student_hall_stereo.py correct
- stereo_calibration: failed, 4.58s, 종료 1; .venv/bin/python scripts/refine_student_hall_stereo.py calibrate
- stereo_calibration_columnT: succeeded, 3.80s, 종료 0; .venv/bin/python scripts/refine_student_hall_stereo.py calibrate
- stereo_calibration_metric: succeeded, 3.82s, 종료 0; .venv/bin/python scripts/refine_student_hall_stereo.py calibrate
- stereo_depth_all: succeeded, 17.09s, 종료 0; .venv/bin/python scripts/refine_student_hall_stereo.py depths
- stereo_gs_prepare: succeeded, 6.70s, 종료 0; .venv/bin/python scripts/refine_student_hall_gs3d.py prepare
- stereo_gs_train_16000: succeeded, 249.63s, 종료 0; .venv/bin/python scripts/refine_student_hall_gs3d.py train --steps 16000
- stereo_multiview: succeeded, 5.93s, 종료 0; .venv/bin/python scripts/refine_student_hall_stereo.py multiview
- stereo_multiview_corrected: succeeded, 6.14s, 종료 0; .venv/bin/python scripts/refine_student_hall_stereo.py multiview
- stereo_sfm_features: succeeded, 20.29s, 종료 0; .venv/bin/python scripts/refine_student_hall_colmap.py features
- stereo_sfm_lidar_alignment: succeeded, 2.21s, 종료 0; .venv/bin/python scripts/refine_student_hall_alignment.py
- stereo_sfm_mapping: succeeded, 31.63s, 종료 0; .venv/bin/python scripts/refine_student_hall_colmap.py mapping
- stereo_sfm_matching: succeeded, 454.05s, 종료 0; .venv/bin/python scripts/refine_student_hall_colmap.py match
- stereo_sfm_prepare: succeeded, 3.28s, 종료 0; .venv/bin/python scripts/refine_student_hall_colmap.py prepare
- surface_visibility_color: succeeded, 19.09s, 종료 0; .venv-mesh312/bin/python scripts/refine_student_hall_surface.py --recolor --final
- surfel_initialization: succeeded, 0.95s, 종료 0; .venv/bin/python scripts/refine_student_hall_2dgs.py --prepare

## 사용한 구현과 참고
- Metal 2DGS implementation: https://github.com/tchauffi/metalsplat
- Native Metal 3DGS implementation: https://github.com/rayanht/msplat
- COLMAP: https://colmap.github.io/
- 2D Gaussian Splatting reference: https://github.com/hbb1/2d-gaussian-splatting
