#!/usr/bin/env python3
from pathlib import Path
import json

ROOT=Path('experiments/student_hall_lidar');F=ROOT/'fused'

def main():
    s=json.loads((F/'summary.json').read_text());ev=json.loads((ROOT/'evaluation.json').read_text());j=json.loads((ROOT/'joint_report.json').read_text())
    p=json.loads((F/'poisson_validation.json').read_text());t=json.loads((F/'mesh_validation.json').read_text())
    trials=[]
    for path in ROOT.glob('gs_*_*/metrics.json'):
        trials.append(json.loads(path.read_text()))
    trials.sort(key=lambda x:(x['method'],x['iterations']));best=max(trials,key=lambda x:x['metrics']['psnr'])
    names={'kiss':'LiDAR 단독 KISS-ICP','colmap_metric':'원본 COLMAP + LiDAR 크기 보정','icp_odometry':'시각 초기값·전역 정합·ICP','posegraph':'전역 정합 + 포즈 그래프 (구조 선택)','joint':'COLMAP/LiDAR 공동 보정','sensor_clean':'근거리 제외 후 정합 재보정 (추가 후보)'}
    table='\n'.join(f"| {names[r['method']]} | {r['frames']} | {r['overlap_25cm']*100:.2f}% | {r['plane_median_matched_cm']:.3f} cm | {r['clipped_plane_mean_cm']:.3f} cm |" for r in ev['metrics'])
    gs='\n'.join(f"| {x['method']} | {x['iterations']:,} | {x['metrics']['psnr']:.3f} | {x['metrics']['ssim']:.4f} | {x['final_gaussians']:,} |" for x in trials)
    meshes='\n'.join(f"| TSDF {x['voxel_m']:.2f}m | {x['clipped_mae_cm']:.2f} cm |" for x in t['candidates'])
    meshes+='\n'+'\n'.join(f"| Poisson depth {x['depth']}, 근거리 제외 | {x['clipped_mae_cm']:.2f} cm |" for x in p['candidates'])
    report=f'''# 학생회관 RGB/COLMAP + LiDAR 복원 실험

2026-10-21 KST. 실제 파일을 생성·비교한 결과입니다.

공유 페이지: https://stereo-reconstruction-lab-kyunghwan.sillycod2.chatgpt.site/student-hall.html

## 결과와 선택

- 입력은 `01-26-15-41-24`의 RGB 사진 48개와 원시 LiDAR 스캔 48개입니다. 조형물이 있는 중정에서 계단·경사로를 거쳐 학생회관 표지판이 있는 입구까지 이동합니다.
- 친구의 KISS-ICP 완성 지도/궤적은 받지 않았습니다. 이 프로젝트에 있던 같은 촬영의 `storage.hdf5`에서 LiDAR 지도를 직접 만들었습니다.
- 원본 COLMAP 두 컴포넌트는 중복을 제외해 40개 카메라, 합계 5,183개 특징점입니다. 각 컴포넌트를 미터 단위로 보정하고 공통 카메라로 배치해 비교했습니다. 예전 임시 경로 합본은 정밀 정합의 입력으로 사용하지 않았습니다.
- 새 지도는 48개 촬영 시점을 연결하고, **컬러 점 {s['colored_points']:,}개**, **전체 LiDAR 점 {s['full_map_points']:,}개**를 포함합니다. 컬러 점 수는 특징점 수의 약 {s['colored_points']/5183:.1f}배지만, 점 수의 증가가 정확도 향상 배수를 뜻하지 않습니다.
- 선택한 Poisson 표면은 원본 **삼각형 {s['mesh_triangles']:,}개**입니다. 사이트는 영역별로 최대 18만 개로 줄인 표시용 메시를 제공합니다. TSDF 표면 {s['tsdf_mesh_triangles']:,}개도 별도로 보관했습니다.
- 구조는 `posegraph`를, 렌더링은 `{best['method']}` {best['iterations']:,}회 모델을 선택했습니다. LiDAR 초기화가 렌더링까지 개선된다고 확인되지는 않았습니다.

## 실제로 시도한 방법

1. **LiDAR 깊이를 이용한 COLMAP 크기 보정.** 원본 카메라 관측과 LiDAR 깊이의 비율 중앙값을 사용했습니다. 컴포넌트별 상대 깊이 차이 중앙값은 초기 약 2.67%, 2.30%입니다. 이 수치도 실측 정답 오차는 아닙니다.
2. **KISS-ICP 1.3.0 단독 추적.** 별도 시점별 타임스탬프가 없어 스캔 내 motion deskew는 적용하지 않았습니다. 저장된 스캔 간 간격도 수 초여서 일반적인 고속 연속 LiDAR 입력보다 어렵습니다.
3. **사진 특징점 + LiDAR 깊이 PnP, COLMAP 상대 위치, KISS 초기값 비교.** 각 후보를 여러 해상도의 robust point-to-plane ICP로 보정했습니다.
4. **FPFH + RANSAC 전역 정합으로 어려운 회전 복구.** 예를 들어 프레임 17→18은 초기 약 33.5%였던 정합 겹침이 약 64.2%로 높아졌습니다. 이 값은 해당 정합의 적합도이며 아래 별도 반환점 평가와 다릅니다.
5. **포즈 그래프 최적화.** 인접·건너뛴 시점의 검증된 LiDAR 정합 관계를 함께 최적화했습니다.
6. **COLMAP 카메라·특징점 + LiDAR 접평면 공동 보정.** {j['points']:,}개 특징점, {j['observations']:,}개 영상 관측과 {j['lidar_plane_constraints']:,}개 평면 제약을 함께 사용했습니다. 별도로 남긴 관측 {j['reserved_observations']:,}개의 재투영 오차 중앙값은 {j['before_median_px']:.3f}→{j['after_median_px']:.3f}px, 90백분위는 {j['before_p90_px']:.3f}→{j['after_p90_px']:.3f}px입니다. 최적화는 ftol 조건으로 종료했습니다. 이 관측은 이번 보정에서 제외했지만, 최초 COLMAP 추정에서는 사용됐습니다. 새로운 사진 일반화 점수가 아닙니다.
7. **TSDF와 Poisson 표면 비교.** Poisson 초기 후보의 거짓 면을 실제 시점 렌더링에서 발견했고, 센서 가까이 3m 이내 반환점 제외 및 실측 점으로부터 25cm 이상 떨어진 정점 제거를 적용했습니다. 이후 영상으로 다시 확인했습니다. 회색·구멍·식생 주변 덩어리와 일부 부정확한 면은 남아 있습니다.
8. **동일한 사진 분할의 Gaussian Splatting 비교.** COLMAP 원본, 공동 보정한 카메라/특징점, LiDAR 점 초기화를 각각 학습했습니다. RGB 손실을 사용했습니다. **LiDAR 깊이 손실이나 LI-GS 논문 전체를 구현한 실험은 아닙니다.**

이 구현은 Colmap-PCD의 문제 설정과 기존 정합·최적화 기법을 참고한 프로젝트용 파이프라인입니다. Colmap-PCD 공식 실행 파일을 설치·재현했다고 주장하지 않습니다.

## 같은 구간에서 비교한 스캔 일관성

원본 COLMAP에서 연결된 것과 동일한 37개 인접 카메라 쌍을 모든 후보에 사용했습니다. ICP에는 배열의 짝수 반환점을, 아래 검사에는 홀수 반환점 {ev['metrics'][0]['heldout_returns']:,}개를 사용했습니다. 최종 컬러 지도와 메시에는 더 많은 원시 관측을 사용합니다.

| 방법 | 카메라 수 | 25cm 이내 겹침 ↑ | 대응점 표면 오차 중앙값 ↓ | 50cm로 제한한 표면 오차 평균 ↓ |
|---|---:|---:|---:|---:|
{table}

표면 오차는 대상 스캔의 접평면까지 거리입니다. 대응점 중앙값은 거리 25cm 이내 대응에 한하므로, 겹침 비율과 함께 읽어야 합니다. 평균은 전체 검사점의 평면 거리를 50cm에서 제한했습니다. 가장 작은 값인 순차 ICP와 포즈 그래프는 매우 가깝습니다. 구조의 최종 결과는 여러 시점 제약을 사용하는 포즈 그래프를 채택했습니다.

공동 보정은 영상 재투영을 개선했지만 스캔 일관성은 떨어졌습니다. 따라서 구조 모델에는 사용하지 않았고, 해당 카메라는 외관 학습 비교에 사용했습니다.

## 표면 비교

검증용 표면을 만들 때 프레임 5, 13, 21, 29, 37, 45의 깊이/스캔을 제외했습니다. 최종 선택 후 모든 스캔으로 표면을 다시 만들었습니다. 포즈 추정에는 전체 촬영을 사용했습니다.

| 표면 후보 | 제외한 여섯 시점의 깊이 오차/누락 페널티 평균 ↓ |
|---|---:|
{meshes}

오차는 1m에서 제한하고 표면이 없는 광선에는 1m 페널티를 줬습니다. 따라서 이 값은 누락 영역을 포함하는 비교 지표이며, 건물이 이 센티미터 수치로 정확하다는 뜻이 아닙니다. LiDAR 깊이도 제한된 삼각형 보간을 사용합니다. 학생회관 계단·입구에 해당하는 프레임 21,29,37,45를 중심으로 Poisson depth {p['selected_depth']}를 선택했습니다. 초기의 depth 8/9, 근거리 제거 전 depth 10/11 결과와 실패 로그도 보존했습니다.

## 새로운 시점의 렌더링

동일한 프레임 20~45 중 21개는 외관 학습, 5개(20,26,32,38,44)는 평가에 사용했습니다. 모든 카메라는 전체 사진을 사용해 추정했습니다. 초기 LiDAR 깊이·색상에는 외관 평가 사진을 제외했습니다. 원본 COLMAP 특징점 자체는 최초 전체 SfM에서 나온 결과입니다.

| 초기화/카메라 | 반복 수 | 평가 PSNR ↑ | 평가 SSIM ↑ | Gaussian 수 |
|---|---:|---:|---:|---:|
{gs}

`colmap`은 원래 COLMAP 카메라/점, `sparse`는 공동 보정한 카메라/COLMAP 점, `lidar`는 공동 보정한 카메라/LiDAR 점입니다. `sparse` 대 `lidar`는 동일 카메라·학습 설정에서 초기 점만 바뀝니다. `colmap`과 비교하면 카메라도 달라집니다. 난수까지 통제한 다중 반복 통계 실험은 아닙니다.

선택 모델의 PSNR은 **{best['metrics']['psnr']:.3f}dB**, SSIM은 **{best['metrics']['ssim']:.4f}**입니다. 이전 09-05 경로의 26.935dB와는 장면·사진·분할이 달라 직접 비교할 수 없습니다. LiDAR를 단순히 초기 점으로 넣은 모델은 더 나빴습니다. 이후 LiDAR 깊이 손실 구현을 검토할 근거는 있지만, 현재 결과가 그 효과를 증명하지는 않습니다.

## 파일과 재실행

- 전체 LiDAR 지도: `experiments/student_hall_lidar/fused/lidar_full_map.ply`
- 컬러 점군: `experiments/student_hall_lidar/fused/student_hall_colored.ply`
- 선택한 원본 메시: `experiments/student_hall_lidar/fused/student_hall_poisson.ply`
- 직접 깊이를 연결한 메시: `experiments/student_hall_lidar/fused/student_hall_tsdf.ply`
- 구조 포즈: `experiments/student_hall_lidar/posegraph_poses.npz`
- 공동 최적화 포즈·점: `experiments/student_hall_lidar/joint_poses.npz`
- 비교 지표: `evaluation.json`, `joint_report.json`, `fused/mesh_validation.json`, `fused/poisson_validation.json`
- 렌더링: `experiments/student_hall_lidar/gs_{best['method']}_{best['iterations']}/`
- 사이트: `sites/reconstruction-lab-share/dist/student-hall.html`

```sh
.venv/bin/python -m pip install kiss-icp==1.3.0
.venv/bin/python scripts/reconstruct_student_hall_lidar.py prepare
.venv/bin/python scripts/reconstruct_student_hall_lidar.py sfm
.venv/bin/python scripts/reconstruct_student_hall_lidar.py kiss
OMP_NUM_THREADS=4 .venv/bin/python scripts/reconstruct_student_hall_lidar.py register
OMP_NUM_THREADS=4 .venv/bin/python scripts/reconstruct_student_hall_lidar.py rescue
OMP_NUM_THREADS=4 .venv/bin/python scripts/reconstruct_student_hall_lidar.py register
OMP_NUM_THREADS=4 .venv/bin/python scripts/reconstruct_student_hall_lidar.py joint
OMP_NUM_THREADS=4 .venv/bin/python scripts/reconstruct_student_hall_lidar.py evaluate
OMP_NUM_THREADS=4 .venv/bin/python scripts/reconstruct_student_hall_lidar.py export
OMP_NUM_THREADS=4 .venv-mesh312/bin/python scripts/mesh_student_hall_lidar.py
OMP_NUM_THREADS=4 .venv-mesh312/bin/python scripts/poisson_student_hall.py
.venv/bin/python scripts/student_hall_gaussians.py prepare
.venv/bin/python scripts/student_hall_gaussians.py sparse --iterations 12000
.venv/bin/python scripts/student_hall_gaussians.py lidar --iterations 12000
.venv/bin/python scripts/student_hall_gaussians.py colmap --iterations 12000
.venv/bin/python scripts/student_hall_gaussians.py colmap --iterations 30000
.venv/bin/python scripts/write_student_hall_report.py
OMP_NUM_THREADS=4 .venv/bin/python scripts/prepare_student_hall_site.py geometry
.venv/bin/python scripts/prepare_student_hall_site.py rendering
.venv/bin/python scripts/verify_student_hall_results.py
```

GPU 학습에는 Mac Metal 장치 접근이 필요합니다. Open3D 0.20 또는 `linear_fit=True`의 Poisson 후보는 native 오류로 실패해 Python 3.12/Open3D 0.19, `linear_fit=False`, 단일 meshing thread 조합을 사용했습니다. 최종 실행에도 라이브러리 경고가 일부 있었으므로 유한 좌표·인덱스 검사와 실제 기하 렌더링을 확인했습니다. 최적화와 표면 방식은 입력 데이터에 맞춘 선택이며 범용적으로 최고 성능이라고 보장하지 않습니다.

## 검증과 한계

- 3D 구조를 실제 세 카메라 시점으로 ray casting하여 검토했습니다. 초기 가짜 차폐 면은 근거리 반환점 제외 후 사라진 것을 확인했습니다.
- 사이트의 파일·링크·기하 버퍼·동영상·JavaScript 구문을 검사합니다. 현재 세션에는 사용 가능한 브라우저가 없어 실제 브라우저의 마우스/터치 조작과 화면 배치는 검증하지 못했습니다.
- 건물 전체의 외벽, 실내, 지붕을 빠짐없이 복원한 디지털 트윈이 아닙니다. 사진이 보지 않은 표면은 회색이고, 빈 곳·잔여 잡음·부정확한 얇은 구조가 남습니다.
- 따로 측정한 건물 치수나 측량 기준점이 없으므로 절대 위치·크기 오차는 확정할 수 없습니다. LiDAR 역시 ground truth로 간주하지 않았습니다.

참고: [Colmap-PCD](https://arxiv.org/abs/2310.05504), [KISS-ICP](https://github.com/PRBonn/kiss-icp), [Open3D multiway registration](https://www.open3d.org/docs/release/tutorial/pipelines/multiway_registration.html), [LI-GS](https://changjianjiang01.github.io/LI-GS/).
원본 데이터: [Pixel-aligned RGB-NIR Stereo, CVPR 2025](https://github.com/divisonofficer/Pixel_aligned_RGB_NIR_Stereo), Jinnyeong Kim · Seung-Hwan Baek / POSTECH.
'''
    Path('STUDENT_HALL_LIDAR_REPORT.md').write_text(report)
    (Path('sites/reconstruction-lab-share/dist')/'student-hall-report.md').write_text(report)
    print('report_written',len(report),'selected_render',best['method'],best['iterations'])

if __name__=='__main__':main()
