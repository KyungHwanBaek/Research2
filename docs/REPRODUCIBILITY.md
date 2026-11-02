# 환경과 재현 안내

이 저장소는 실험 코드와 보고서를 보존한다. 제외된 원본 데이터, SfM 카메라 파일, 깊이·포즈 중간 산출물, 외부 라이브러리, 체크포인트가 필요하므로 전체 최종 결과의 한 명령 재현을 보장하지 않는다. 이번 GitHub 정리는 복원을 새로 실행한 것이 아니다.

## 환경

원 실행 환경은 Apple M4, 24GB, macOS arm64다. 초기 COLMAP은 Homebrew 4.2.1_1 CPU 빌드, `pycolmap==4.2.1`을 사용했다. 일반 작업은 Python 3.13 가상환경, 메시 작업은 Python 3.12/Open3D 0.19 별도 환경에서 진행했다. 테스트 환경에서 Open3D 0.20 및 일부 Poisson 설정은 실패해 0.19를 사용했다.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

# Python 3.12가 설치되어 있어야 함
python3.12 -m venv .venv-mesh312
.venv-mesh312/bin/python -m pip install -r requirements.txt -r requirements-stereo-mesh.txt
```

LiDAR 작업의 추가 의존성은 `requirements-lidar.txt`에 있다. 일부 Gaussian·학습 깊이 후보는 별도 외부 코드와 환경을 요구하므로 이 두 환경만으로 모든 스크립트가 실행되지는 않는다. 의존성 파일에는 최소 버전도 포함되어 있어 완전한 lockfile은 아니다.

## 원본 데이터와 초기 SfM

공식 데이터의 사용 조건을 확인한 뒤 필요한 시퀀스만 다운로드한다. 아래는 새 데이터 준비·초기 SfM의 진입점이며 최종 스테레오/OpenMVS 결과 재현 명령과는 구분한다.

```bash
.venv/bin/python scripts/download_dataset.py --filename 01-26-15-41-24.tar.gz
.venv/bin/python scripts/extract_archive.py data/raw/01-26-15-41-24.tar.gz data/extracted/01-26-15-41-24
.venv/bin/python scripts/prepare_colmap_images.py --help
bash scripts/run_colmap_sparse.sh <prepared-image-directory> <experiment-directory> <fx,fy,cx,cy>
```

다운로드·추출 스크립트는 기존 경로 덮어쓰기를 피한다. 실제 아카이브 구조와 보정 파일을 확인하고 사진 준비 옵션을 정한다. 캘리브레이션 값은 원 데이터와 사진 해상도에 맞춰야 한다.

## 최종 학생회관 단계별 입력 조건

| 단계 | 코드 | 필요한 선행 산출물 |
|---|---|---|
| LiDAR 정합 | `reconstruct_student_hall_lidar.py` | 원본 HDF5·보정·사진·SfM 모델 |
| LiDAR 표면 | `poisson_student_hall.py` | `experiments/student_hall_lidar/frames/*.npz`, `fused/selected_poses.npz`, `fused/lidar_full_map.ply` |
| 사진 색상 투영 | `refine_student_hall_surface.py --recolor --final` | 기존 Poisson 메시·사진·카메라 정보 |
| OpenMVS | `hall_geometry_mvs.py` | 스테레오 SfM, 좌우 사진, 기존 LiDAR 카메라 위치, OpenMVS 실행 파일 |
| 사진 정점 보정·텍스처 | `hall_geometry_finish.py` | OpenMVS의 `scene_mesh.mvs`와 원본 사진 |
| 좌표 변환·파일 내보내기 | `hall_geometry_surface.py`, `hall_geometry_package.py` | 최종 메시·기록된 좌표 변환·평가 결과 |

`hall_geometry_mvs.py`는 `prepare`, `import`, `dense`, `mesh` 단계와 `--variant stereo --mode final` 등을 제공한다. 최종 기록의 variant와 원본 입력을 복원한 뒤 순서대로 실행해야 한다. `hall_geometry_finish.py`는 `refine --mode final`과 `texture --refined` 단계를 제공한다. 후속 패키지 스크립트는 다른 후보의 평가 파일도 읽을 수 있어 단독 실행용은 아니다.

OpenMVS 실행 파일은 `third_party/openmvs/{InterfaceCOLMAP,DensifyPointCloud,ReconstructMesh,RefineMesh,TextureMesh}` 경로를 기대한다. 공식 소스는 `docs/third_party_versions.json`의 revision을 참고하고 CPU 빌드를 준비한다. 일부 스크립트는 작업 폴더 또는 과거 실험 경로를 가정하므로 실행 전 경로를 확인한다.

## 스크립트 분류

- `hall_geometry_*`: 사진 형상·깊이·메시 실험
- `refine_student_hall_*`, `hall_followup_*`: 표면·SfM·색상·Gaussian 후속 후보
- `hall_research_*`: Metal Gaussian 환경 변경·추가 학습·외관 평가
- `prepare_*site.py`, `*publish.py`, `build_hall_overview.py`: 별도 사이트·큰 자산을 필요로 하는 배포/표시 준비
- `create_*docx.py`: 문서 생성용이며 복원 알고리즘과 별개

## 검증 범위

GitHub 업로드 전에는 Python 구문, 셸·JS 문법과 파일 크기·인증정보 포함 여부를 확인한다. 데이터가 제외된 깨끗한 clone에서 전체 복원을 재실행했다는 뜻은 아니다. 실제 과거 실행 결과·실패·선택 조건은 해당 보고서를 읽는다.
