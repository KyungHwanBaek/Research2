# 학생회관 사진 기반 3D 복원 실험

여러 시점의 RGB 사진에서 학생회관 계단·입구·벽의 **실제 3D 표면**을 만드는 프로젝트입니다. COLMAP, LiDAR, OpenMVS와 Mac용 Gaussian Splatting을 적용·비교한 코드와 실험 기록을 정리했습니다.

**현재 결과:** 사진 기반 점군·메시·텍스처 GLB를 만들었습니다. 기존 LiDAR 기반 표면보다 구조적으로 정확하다는 근거는 확보하지 못했습니다. 새로운 알고리즘이나 공개 벤치마크 최고 성능을 주장하지 않습니다.

[**실제 결과 ①~④ 비교하기**](https://stereo-reconstruction-lab-kyunghwan.qj8scj68m5.chatgpt.site/student-hall-overview#viewer-section)

## 먼저 읽을 문서

- [전체 작업 요약 / 다른 LLM에 전달할 문서](docs/PROJECT_SUMMARY_KO.md)
- [환경·데이터·단계별 재현 안내](docs/REPRODUCIBILITY.md)
- [외부 코드·데이터 출처와 포함 범위](docs/SOURCES_AND_SCOPE.md)
- [사진 기반 최종 형상 보고서](reports/PHOTO_GEOMETRY_20261031.md)
- [Metal Gaussian 추가 실험 보고서](reports/GAUSSIAN_RESEARCH_20261027.md)
- [초기 실험 전체 기록](docs/archive/README_20261021.md)

## 네 가지 결과

| 결과 | 방법 | 원본 산출물 |
|---|---|---|
| ① 초기 COLMAP | 사진 대응 → 카메라 정보·희소 특징점 | 초기 비교 점군 5,183개 점 |
| ② 누적 LiDAR | 48개 스캔을 사진 위치 정보와 기하 정합을 참고해 누적 | 약 164.8만 개 측정점 |
| ③ COLMAP + LiDAR | LiDAR 점·법선 → Poisson 표면 → RGB 색상 투영 | 2,120,479개 삼각형 |
| ④ COLMAP → OpenMVS | RGB 좌우 52장 → 깊이·점군·메시·정점 보정·텍스처 | 317,561개 점 / 최종 435,398개 삼각형 |

③은 COLMAP 위치, LiDAR 형상, RGB 색상을 함께 활용합니다. ④의 표면은 사진에서 계산하며 LiDAR는 좌표·크기 정렬에 사용합니다. 초기 COLMAP 비교 사본도 LiDAR 기준으로 좌표·크기를 맞췄습니다. 웹 화면은 관심 영역과 점·삼각형 수를 줄인 표시본입니다.

같은 학생회관 촬영 구간을 복원했습니다. 별도로 조사한 다른 실내·실외 경로는 신뢰할 연결을 얻지 못해 최종 모델에 합치지 않았습니다.

## 주요 코드

모든 스크립트는 `scripts/`에 있습니다.

| 단계 | 진입점 |
|---|---|
| 데이터·사진 준비 | `download_dataset.py`, `extract_archive.py`, `prepare_colmap_images.py` |
| 초기 SfM | `run_colmap_sparse.sh`, `summarize_colmap.py` |
| LiDAR 정합·표면 | `reconstruct_student_hall_lidar.py`, `mesh_student_hall_lidar.py`, `poisson_student_hall.py` |
| 사진 색상 투영 | `refine_student_hall_surface.py` |
| OpenMVS 깊이·메시 | `hall_geometry_mvs.py` |
| 사진 정점 보정·텍스처 | `hall_geometry_finish.py` |
| Metal Gaussian 실험 | `train_msplat_render_flythrough.py`, `hall_research_experiments.py` |
| 구조 진단·내보내기·표시 | `hall_geometry_surface.py`, `hall_geometry_package.py`, `build_hall_overview.py` |

파일명에 특정 방법이 포함되어 있어도 해당 논문의 공식 알고리즘을 완전히 재현했다는 의미는 아닙니다. 실제 실행 범위·실패·채택 결과는 보고서가 기준입니다.

## 환경과 재현 범위

- Apple M4, 통합 메모리 24GB, macOS arm64
- CPU 기반 COLMAP·OpenMVS, Python 3.12/Open3D 0.19 메시 처리
- Metal 기반 `msplat` 실행 기록 보유, NVIDIA CUDA 없음
- `requirements.txt`, `requirements-lidar.txt`, `requirements-stereo-mesh.txt`에 단계별 의존성 기록

이 저장소는 **실험 코드와 기록의 공개본**입니다. 원본 데이터·카메라/깊이 중간 산출물·외부 라이브러리·모델 가중치를 제외하므로 clone 후 최종 결과가 한 명령으로 재현되는 패키지는 아닙니다. [재현 안내](docs/REPRODUCIBILITY.md)를 먼저 확인하세요.

## 평가와 후속 아이디어

일부 LiDAR 스캔을 표면 생성에서 제외한 진단을 수행했으나 위치 추정에는 전체 자료가 관여했고 같은 검증을 반복 사용했습니다. 외관 PSNR·SSIM과 건물 형상 정확도는 구분합니다.

ETH3D·Tanks and Temples·DTU MVS 공식 평가는 **미실행**입니다. 관측 신뢰도와 정렬 변화에 대한 안정성으로 사진 점을 선택하는 결합 방식도 **후속 제안이며 미구현**입니다. 독창성은 선행연구 조사와 비교 실험으로 추가 검증해야 합니다.

## 원 데이터

Jinnyeong Kim, Seung-Hwan Baek, *Pixel-aligned RGB-NIR Stereo Imaging and Dataset for Robot Vision*, CVPR 2025.

- [공식 프로젝트](https://divisonofficer.github.io/project_page_Pixel_aligned_RGB_NIR_Stereo/)
- [공식 데이터](https://huggingface.co/datasets/DivisonOfficer/Pixel-aligned_RGB-NIR_stereo_dataset)

외부 소스·데이터는 재배포하지 않습니다. 사용 시 각 원저작자의 라이선스·인용 조건을 확인하세요.
