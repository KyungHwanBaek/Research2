# 학생회관 추가 연구 및 직접 실행 보고서 · 2026-10-27

## 결론
이전 선택 모델 유지: PSNR 24.6562dB, SSIM 0.7946. 추가 후보는 채택 기준을 넘지 못했다.
실행은 성공/실패를 모두 기록했다. 품질이 나아지지 않은 후보도 삭제하지 않았다. 물리적 LiDAR mesh는 이번 라운드에서 변경하지 않았다.

## 사용 자료와 비교 조건
이전과 같은 학생회관 RGB 좌우 52장 중, 검증 시점 [20,26,32,38,44]의 양쪽 사진 10장은 외관 학습에서 제외했다. 학습 42장, 모델 선택용 검증은 왼쪽 5장이다. 별도 오른쪽 5장은 선택 후 진단에만 사용했다. 원시 LiDAR 48개와 기존 Poisson mesh는 보존했으며 이번 RGB 학습에 LiDAR 깊이 손실을 새로 넣지는 않았다. 친구의 별도 지도, 새 촬영, NIR 학습은 없다.
기존 선택 모델은 freeze_principal_960, step5000, Gaussian 211,669개. 이 체크포인트에서 각 후보를 독립적으로 시작했다. 전체 SfM 추정은 검증 사진을 포함하므로 독립 시험이 아니다. 여러 차례 같은 왼쪽 검증을 재사용한 선택 편향도 있다.

## 왜 이 방법을 골랐나
이전 결과에서 바닥·난간·나무 주변 번짐이 남았고 opacity reset과 무분별한 점 증가가 화질을 낮추는 구간이 있었다. 논문/공식 코드에서 찾은 다음 후보는 카메라별 색상 차이, 샘플링·aliasing, 세부 표현을 위한 분할, 깊이와 표면 제약이었다. 먼저 현재 Metal 장비에서 직접 검증할 수 있는 세 가지 변경과 처리 한도 진단을 실행했다. 논문 전체를 재현한 것으로 부르지 않는다.

## 1. 렌더러 처리 한도 진단
기존 native 로그의 타일당 2048개 한도 경고를 확인했다. 원래 설치본을 수정하지 않고 소스를 복제해 한도를 4096개로 올리고, 마지막 렌더링의 overflow 플래그와 학습률 조절 API를 추가했다. 4096개 sort가 사용하는 threadgroup 배열은 32KiB다. 원래 2048개의 2배이며 무제한 렌더러가 아니다.
첫 오프라인 빌드는 xcrun metal 실행 파일이 없어 실패했다. macOS Metal 런타임의 newLibraryWithSource로 MSL을 컴파일하는 별도 빌드를 만들었고 성공했다. 실패와 수정 patch를 모두 남겼다. 런타임/컴파일러 변경도 함께 있으므로 미세한 픽셀 차이의 원인을 한도에만 돌리지 않는다.
같은 모델 재렌더링의 평균 차이는 약 +0.0002dB로, 실질 개선으로 채택하지 않았다. 검증 5장 모두 4096 한도 overflow가 없었다. 이것이 모든 촬영·모든 향후 모델에 overflow가 없다는 보장은 아니다.

## 2. 실제 원본 해상도에서 저학습률 추가 학습
기존 왼쪽 학습에는 960×720 사본이 사용되었고 1440×1080 왜곡 보정 원본도 남아 있었다. 이번에는 좌우 모두 1440 원본에서 준비했다. 원본을 960으로 줄인 대조군과 원본 1440 학습을 각각 4,000회 실행했다. 실제 추가 정보가 없는 960→1440 확대와 구분했다.
학습률은 기존 6개 Gaussian 파라미터 그룹과 mean scheduler를 0.25배로 줄였다. 점 수는 고정하고 위치·크기·회전·색상·불투명도는 계속 조정했다. 500회마다 같은 검증으로 확인해 가장 좋은 체크포인트를 보존했다. 960/1440 렌더링은 area downsample로 720×540으로 맞춰 같은 원본 검증 이미지와 비교했다. 사진 전처리 및 pixel footprint가 함께 바뀌므로 해상도 단독 효과로 단정하지 않는다. 시작점 평가도 남겨 더 학습했을 때의 실제 이득을 구분한다.

## 3. 학습 사진만 사용한 좌우 색상 보정
SfM track에서 왼쪽 학습 사진 2장 이상에 관측된 점을 골랐다. 3×3 RGB patch 중앙값을 취하고, 왼쪽 관측 중앙값에 오른쪽 각 사진을 맞추는 RGB별 affine gain/bias를 robust least squares로 구했다. soft-L1 0.03, gain 범위0.7–1.4, bias ±0.1, identity 정규화를 적용했다. 검증 시점 양쪽 사진은 fitting에도 쓰지 않았다.
이 보정은 사진 전처리이며 공식 3DGS의 학습 중 per-image exposure optimization과 동일한 알고리즘은 아니다. 오른쪽 학습 사진에만 적용해 왼쪽 외관 기준을 유지했다. 이 데이터셋에서 밝기 차이를 줄이는 것이 검증 렌더링까지 개선하는지 별도로 시험했다. patch 색상 오차 감소만으로 3D 품질 개선이라 부르지 않았다.

## 4. 고해상도에서 Gaussian 재분할
큰 Gaussian이 세부 정보를 덮는 가능성을 확인하려고, opacity reset을 끈 채 500회마다 기존 msplat의 표준 signed-gradient densification을 다시 허용했다. threshold0.0003, world scale threshold0.01, screen-size 분할 비활성, split stop9000, scheduler max18000. 4,000회 예산과 Gaussian600,000개 상한을 두었다. 이 후보는 고정 점 대조군과 scheduler도 달라 복합 후보로 해석한다. AbsGS의 absolute-gradient를 구현한 것은 아니다.

## 수치 결과
| 후보 | 보존 모델 PSNR | SSIM | step | Gaussian 수 | 실제 추가 학습 후 최고 PSNR |
|---|---:|---:|---:|---:|---:|
| 이전 선택 모델 | 24.6562 | 0.7946 | 5000 | 211,669 | — |
| 처리 한도 2배 · 재렌더링 | 24.6564 | 0.7946 | 5000 | 211,669 | — |
| 원본→960px · 낮은 학습률 | 24.6564 | 0.7946 | 5000 | 211,669 | 24.6015 |
| 원본 1440px · 낮은 학습률 | 24.5922 | 0.7921 | 5500 | 211,669 | 24.5922 |
| 원본 1440px + 좌우 색상 보정 | 24.5835 | 0.7920 | 5000 | 211,669 | 24.5029 |
| 원본 1440px + 점 재분할 | 24.5835 | 0.7920 | 5000 | 211,669 | 24.2618 |

보존 모델에는 시작 step5000도 포함된다. 따라서 step5000이 최고인 경우 추가 학습 자체가 나아졌다는 뜻이 아니다. 지표는 저장 PNG를 float64로 읽어 계산한 PSNR, Gaussian SSIM(sigma1.5, window11, population covariance)이다. 이전 화면 반올림 24.66dB와 같은 기준이며 재렌더링의 약0.00002dB 수치 차이는 유의미한 개선이 아니다. 채택 기준은 이전 대비 +0.05dB 이상, SSIM 하락0.002 이내다. 이는 통계적 유의성 시험이 아니라 작은 수치 변동에 따른 교체를 피하기 위한 실무 규칙이다.

## 오른쪽 사진 추가 진단
- 이전 선택 모델: PSNR 24.3627dB, SSIM 0.7968, pose reproduction max error 6e-08.
- 원본 1440px · 낮은 학습률: PSNR 24.3253dB, SSIM 0.7969, pose reproduction max error 6e-08.
오른쪽 5장은 외관 학습과 checkpoint 선택에 쓰지 않았다. 정답 사진에 pose나 exposure를 맞추지 않고, 기존 SfM 위치와 오른쪽 내부값으로 렌더링했다. 전체47카메라의 동일 정규화를 재현하고 실제 Dataset train pose와 오차1e-5 미만을 확인했다. 왼쪽 평균에 섞지 않고 별도 표기한다. SfM에는 전체 사진이 사용되어 외부 blind test는 아니다.

## 한계와 다음 실험의 우선순위
이번 추가 실험은 RGB 외관 최적화다. 건물 치수나 LiDAR 구조 정확도가 향상됐다고 주장하지 않는다. 관측되지 않은 바닥, 유리 반사, 가는 난간, 나무의 다른 시점에는 번짐이 남는다. 학습을 늘리거나 원본 해상도를 올리는 것이 항상 좋은 결과를 주지는 않았다.
다음 구조 중심 실험의 우선순위는 (1) SfM–LiDAR 좌표계/센서 시간·외부 보정 오차를 먼저 수치화하고, (2) 정렬 신뢰도가 높은 실제 LiDAR 깊이만 3D Gaussian RGB 학습에 함께 넣고, (3) 독립적으로 남긴 촬영 시점의 raw ray와 사진을 동시에 평가하는 것이다. 현재 깊이를 mesh로 만든 것과 Gaussian 학습에 직접 깊이 제약을 넣는 것은 구분해야 한다.
공식 gsplat의 현재 main에는 LiDAR sensor/rasterization 기능도 있지만 CUDA 기반이다. GPU 환경이 마련되면 같은 사진·카메라·검증 분할로 기본3DGS→antialiased/AbsGrad→depth supervision을 하나씩 비교하는 것이 후속 작업이다. 건물 평면·계단의 surface 자체가 목표면 PGSR을 별도 후보로 비교할 수 있다. 이 장비는 Apple M4이며 CUDA가 없어 이 공식 구현들을 직접 실행했다고 말하지 않는다. 새 클라우드 GPU 비용도 발생시키지 않았다.

## 보존·재현과 배포
원본195개와 이전 선택 모델4개 파일의 SHA-256을 재확인했다. 전체 checkpoint/PLY/splat과 모든 후보는 이 실험 폴더에 남겼다. 공개 ZIP에는 명령·로그·소스·native patch·환경·분할·색상 보정값·평가·비교 그림·입출력 hash를 넣는다. 큰 사진/HDF5/체크포인트는 중복 포함하지 않는다. .splat은 SH 시점별 색상을 단순화한 간편 뷰어용이며, PNG/영상은 전체 학습 모델로 렌더링했다.
브라우저 연결은 사용 가능한 브라우저가 없어 실패했다. 정적 링크·자산·JS와 공개 HTTP·다운로드 검증을 수행하며, 실제 UI 조작은 수행하지 않았다. 배포 영수증과 공개 HTTP 검증은 배포 후 로컬에 따로 저장한다.

## 실제 완료 실행 목록
- native_source_prepare: succeeded, exit 0, 0.05s; .venv/bin/python scripts/hall_research_native.py
- native_isolated_build: failed, exit 1, 13.33s; .venv/bin/python -m pip install --no-deps --target experiments/student_hall_research_20261027/native_package ./experiments/student_hall_research_20261027/native_source
- true_resolution_and_color_prepare: succeeded, exit 0, 12.46s; .venv/bin/python scripts/hall_research_experiments.py prepare
- baseline_render_check: succeeded, exit 0, 3.08s; .venv/bin/python scripts/hall_research_experiments.py train --name baseline_render --engine original --input legacy_960 --steps 0
- native_runtime_prepare: succeeded, exit 0, 0.03s; .venv/bin/python scripts/hall_research_runtime_build.py
- native_runtime_build: succeeded, exit 0, 13.78s; .venv/bin/python -m pip install --no-deps --target experiments/student_hall_research_20261027/native_package ./experiments/student_hall_research_20261027/native_source
- wide_tile_render_check: succeeded, exit 0, 3.95s; .venv/bin/python scripts/hall_research_experiments.py train --name wide_render --engine wide --input legacy_960 --steps 0
- native960_low_lr: succeeded, exit 0, 80.13s; .venv/bin/python scripts/hall_research_experiments.py train --name native960_low_lr --engine wide --input native_960 --steps 4000 --lr .25
- native1440_low_lr: succeeded, exit 0, 137.54s; .venv/bin/python scripts/hall_research_experiments.py train --name native1440_low_lr --engine wide --input native_1440 --steps 4000 --lr .25
- right_probe_prepare: succeeded, exit 0, 0.69s; .venv/bin/python scripts/hall_research_probe.py prepare
- right_probe_baseline: succeeded, exit 0, 2.05s; .venv/bin/python scripts/hall_research_probe.py evaluate --name baseline_render
- native1440_stereo_color: succeeded, exit 0, 139.78s; .venv/bin/python scripts/hall_research_experiments.py train --name native1440_stereo_color --engine wide --input native_1440_color --steps 4000 --lr .25
- native1440_progressive_split: succeeded, exit 0, 138.56s; .venv/bin/python scripts/hall_research_experiments.py train --name native1440_progressive_split --engine wide --input native_1440 --steps 4000 --lr .25 --densify
- evaluate_and_assets: succeeded, exit 0, 3.22s; .venv/bin/python scripts/hall_research_publish.py assets
- right_probe_native1440: succeeded, exit 0, 2.64s; .venv/bin/python scripts/hall_research_probe.py evaluate --name native1440_low_lr
- final_report_page: succeeded, exit 0, 0.45s; .venv/bin/python scripts/hall_research_publish.py report
- scientific_and_asset_qa: succeeded, exit 0, 2.31s; .venv/bin/python scripts/hall_research_check.py
- final_provenance: succeeded, exit 0, 1.75s; .venv/bin/python scripts/hall_research_publish.py provenance
- final_archive_source_refresh: succeeded, exit 0, 1.34s; .venv/bin/python scripts/hall_research_publish.py provenance

## 조사한 논문·공식 구현과 판단
### Official 3D Gaussian Splatting implementation
Official code offers per-image affine exposure optimization, EWA antialiasing, and depth regularization. Depth priors can help floaters and textureless surfaces, but improvement is scene dependent.

판단/실행 범위: Test training-only stereo affine preprocessing locally. It is an adaptation, not a reproduction of learned per-image exposure optimization. Direct 3D depth regularization requires a new differentiable depth path, so is not claimed here.

출처: https://github.com/graphdeco-inria/gaussian-splatting

### Mip-Splatting: Alias-free 3D Gaussian Splatting, CVPR 2024
Combines 3D smoothing with a 2D Mip filter for changing camera sampling rates. Official implementation depends on CUDA.

판단/실행 범위: Do not label simple high-resolution rendering or downsampling as Mip-Splatting. Test actual source-resolution learning and supersampled output separately with the available Metal engine.

출처: https://github.com/autonomousvision/mip-splatting

### AbsGS: Recovering Fine Details for 3D Gaussian Splatting, ACM MM 2024
Analyzes gradient cancellation that can stop large Gaussians from splitting, and uses a homodirectional gradient criterion for densification.

판단/실행 범위: Motivates a new high-resolution standard-densification candidate without opacity resets. This run does not implement AbsGS gradients.

출처: https://arxiv.org/abs/2404.10484

### gsplat official library, current main documentation
CUDA implementation exposes antialiased rasterization and absolute projected gradients. Current main also documents LiDAR rendering, sensor models, and newer appearance compensation. Main changes are distinct from the installed PyPI release.

판단/실행 범위: Best consolidated next backend if NVIDIA CUDA becomes available; not directly executable on this Apple M4 Metal machine. No external GPU spending initiated.

출처: https://github.com/nerfstudio-project/gsplat

### PGSR: Planar-based Gaussian Splatting for Efficient and High-Fidelity Surface Reconstruction, TVCG 2024
Targets geometric surface reconstruction through planar Gaussian rendering. Official installation requires CUDA PyTorch and diff-plane-rasterization.

판단/실행 범위: A plausible surface-focused next experiment, but not installed or claimed as executed here. Requires a CUDA backend or substantial Metal port, with independent LiDAR validation.

출처: https://github.com/zju3dv/PGSR

### Gaussian-LIC2: LiDAR-Inertial-Camera Gaussian Splatting SLAM
Joint LiDAR-camera-inertial odometry and Gaussian mapping; sparse LiDAR depth supervision and depth completion address missing geometry. Uses a continuous-time trajectory model and CUDA acceleration.

판단/실행 범위: Supports prioritizing calibrated LiDAR depth supervision for a future geometry-oriented pipeline. This project currently retains sparse captured scans and estimated poses; no validated synchronized IMU/continuous-time input bundle has been prepared, so a full SLAM reproduction is not claimed.

출처: https://xingxingzuo.github.io/gaussian_lic2/
