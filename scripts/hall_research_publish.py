#!/usr/bin/env python3
"""Publish all candidates, negative findings, primary research and provenance."""
from pathlib import Path
import argparse,datetime,hashlib,html,json,platform,shutil,subprocess,zipfile
import numpy as np
from PIL import Image,ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from hall_research_experiments import O,P,B,VAL,read,dump,metric
ROOT=Path(__file__).resolve().parents[1];SITE=Path('sites/reconstruction-lab-share/dist');A=SITE/'assets'
NAMES=['wide_render','native960_low_lr','native1440_low_lr','native1440_stereo_color','native1440_progressive_split']
LABELS={'baseline_render':'이전 선택 모델','wide_render':'처리 한도 2배 · 재렌더링','native960_low_lr':'원본→960px · 낮은 학습률','native1440_low_lr':'원본 1440px · 낮은 학습률','native1440_stereo_color':'원본 1440px + 좌우 색상 보정','native1440_progressive_split':'원본 1440px + 점 재분할'}
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def score(folder):
    rows=[]
    for k,frame in enumerate(VAL):
        x=np.asarray(Image.open(folder/f'heldout_{k:02d}.png')).astype(float)/255;y=np.asarray(Image.open(B/'gs_images'/f'{frame:06d}.png')).astype(float)/255;assert x.shape==y.shape==(540,720,3);rows.append(dict(frame=frame,**metric(x,y)))
    return dict(psnr=float(np.mean([r['psnr'] for r in rows])),ssim=float(np.mean([r['ssim'] for r in rows])),views=rows)
def evaluate():
    base=dict(name='baseline_render',label=LABELS['baseline_render'],**score(O/'baseline_render'),gaussians=211669,step=5000)
    rows=[]
    for name in NAMES:
        m=read(O/name/'best_metrics.json');h=read(O/name/'history.json');trained=[r for r in h if r['step']>5000]
        rows.append(dict(name=name,label=LABELS[name],**score(O/name),gaussians=m['gaussians'],step=m['step'],best_after_training=max(trained,key=lambda x:x['psnr']) if trained else None,last=h[-1],validation_overflow_flags=m['validation_overflow_flags']))
    good=[r for r in rows if r['psnr']>=base['psnr']+.05 and r['ssim']>=base['ssim']-.002]
    chosen=max(good,key=lambda r:r['psnr']) if good else base
    # Keep model selection confined to the same left validation; right scores follow later.
    result=dict(date='2026-10-27',baseline=base,candidates=rows,selected=chosen,adopted=chosen['name']!='baseline_render',selection_rule='Same five left validation PNGs, 720x540, float64 PSNR and Gaussian SSIM. Adopt only >=+0.05dB and SSIM decline <=0.002. No right diagnostic scores used for checkpoint/candidate selection.',physical_geometry='Unchanged. These runs train RGB/SSIM appearance and do not add LiDAR depth supervision.',caveats=['Repeatedly reused validation; all photographs used in SfM. Not an independent test.','Runtime MSL compilation is an additional difference from the installed offline-compiled original renderer.','Original-resolution input preprocessing and rasterization footprint change together when resolution changes.','No full Mip-Splatting, AbsGS, PGSR, or Gaussian-LIC2 implementation was run.'])
    dump(O/'results.json',result);print('SELECTION',chosen['name'],chosen['psnr'],chosen['ssim'],flush=True);return result
def assets():
    r=evaluate();folder=O/r['selected']['name'];trained_saved=[x for x in r['candidates'] if x['step']>5000];comp=max(trained_saved,key=lambda x:x['psnr']) if trained_saved else r['candidates'][0];candidate=O/comp['name'];dump(O/'comparison_selection.json',dict(name=comp['name'],description='Best saved checkpoint that actually underwent additional training, for visual comparison. Adoption uses the separate 0.05dB rule. Initial-checkpoint fallbacks are explicitly marked in the table.'))
    for k,frame in enumerate(VAL):
        for path,tag in [(B/'gs_images'/f'{frame:06d}.png','input'),(O/'baseline_render'/f'heldout_{k:02d}.png','before'),(candidate/f'heldout_{k:02d}.png','candidate')]:shutil.copyfile(path,A/f'hall-research-{k}-{tag}.png')
    shutil.copyfile(folder/'novel_032.jpg',A/'hall-research-poster.jpg')
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-framerate','12','-i',str(folder/'novel_%03d.jpg'),'-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(A/'hall-research-flythrough.mp4')],check=True)
    if r['adopted']:shutil.copyfile(folder/'best.splat',SITE/'downloads/student_hall_latest.splat')
    # Display every trained checkpoint, not only each retained maximum.
    fig,ax=plt.subplots(figsize=(9,4.5),layout='constrained');ax.axhline(r['baseline']['psnr'],color='#263c30',linestyle='--',label='Previously selected model')
    for name,label in [('native960_low_lr','960px, fixed count'),('native1440_low_lr','1440px, fixed count'),('native1440_stereo_color','1440px, stereo color'),('native1440_progressive_split','1440px, progressive split')]:
        h=read(O/name/'history.json');ax.plot([v['step']-5000 for v in h],[v['psnr'] for v in h],marker='.',label=label)
    ax.set_xlabel('Additional training iterations');ax.set_ylabel('Mean validation PSNR (dB)');ax.spines[['top','right']].set_visible(False);ax.legend(fontsize=8);fig.savefig(O/'learning_curves.png',dpi=180);plt.close(fig);shutil.copyfile(O/'learning_curves.png',A/'hall-research-curves.png')
    sheet=Image.new('RGB',(1440,1800),'white')
    for k,frame in enumerate(VAL):
        for j,tag in enumerate(['input','before','candidate']):
            im=Image.open(A/f'hall-research-{k}-{tag}.png').resize((480,360));ImageDraw.Draw(im).text((10,10),f'{tag} | {frame}',fill='white',stroke_width=2,stroke_fill='black');sheet.paste(im,(j*480,k*360))
    sheet.save(O/'appearance_comparison.jpg',quality=94)
    print('ASSETS_SELECTED',r['selected']['name'],'comparison',comp['name'],flush=True)
def provenance():
    checks=[]
    for row in read(O/'baseline_manifest.json'):
        p=Path(row['path']);ok=sha(p)==row['sha256'];checks.append(dict(path=str(p),unchanged=ok));assert ok
    raw=read(P/'raw_input_manifest.json')
    for row in raw:assert sha(Path(row['path']))==row['sha256'],row['path']
    dump(O/'preservation_checks.json',dict(baseline=checks,raw_inputs_checked=len(raw),all_unchanged=True));shutil.copyfile(P/'raw_input_manifest.json',O/'raw_input_manifest.json')
    snapshot=O/'source_snapshot';snapshot.mkdir(exist_ok=True);sources=list(Path('scripts').glob('hall_research_*.py'))+[Path('scripts/run_hall_research.py')];manifest=[]
    for p in sources:shutil.copyfile(p,snapshot/p.name);manifest.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
    dump(O/'source_manifest.json',manifest);(O/'dependencies.txt').write_text(subprocess.check_output([str(ROOT/'.venv/bin/python'),'-m','pip','freeze'],text=True))
    dump(O/'environment.json',dict(platform=platform.platform(),python=subprocess.check_output([str(ROOT/'.venv/bin/python'),'--version'],text=True).strip(),device='Apple M4, Metal, 24GB; no CUDA',native_isolated=True,browser_ui_test_performed=False))
    outputs=[]
    for name in ['baseline_render']+NAMES:
        for p in [O/name/'best.ply',O/name/'best.splat',O/name/'best_checkpoint']:outputs.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
    dump(O/'output_manifest.json',outputs);print('PROVENANCE',len(raw),len(sources),len(outputs),flush=True)
def report():
    r=read(O/'results.json');base=r['baseline'];selected=r['selected'];adopt=r['adopted'];comparison=read(O/'comparison_selection.json');sources=read(O/'research_sources.json')['sources'];probe={}
    for name in ['baseline_render',selected['name'],comparison['name']]:
        p=O/name/'right_probe_metrics.json'
        if p.exists():probe[name]=read(p)
    r['right_eye_diagnostics']=probe;dump(O/'results.json',r);dump(A/'hall-research-results.json',r)
    status=f"새 후보 {selected['label']} 채택: {base['psnr']:.4f}→{selected['psnr']:.4f}dB" if adopt else f"이전 선택 모델 유지: PSNR {base['psnr']:.4f}dB, SSIM {base['ssim']:.4f}. 추가 후보는 채택 기준을 넘지 못했다."
    table='\n'.join('| '+x['label']+f" | {x['psnr']:.4f} | {x['ssim']:.4f} | {x['step']} | {x['gaussians']:,} | "+(f"{x['best_after_training']['psnr']:.4f}" if x.get('best_after_training') else '—')+' |' for x in [base]+r['candidates'])
    references='\n\n'.join(f"### {s['title']}\n{s['finding']}\n\n판단/실행 범위: {s['decision']}\n\n출처: {s['url']}" for s in sources)
    runs=sorted([read(p) for p in (O/'runs').glob('*.json')],key=lambda x:x['start']);complete=[x for x in runs if x['status']!='running'];runlist='\n'.join(f"- {x['name']}: {x['status']}, exit {x.get('exit_code')}, {x.get('seconds',0):.2f}s; {' '.join(x['command'])}" for x in complete)
    probe_text='\n'.join(f"- {LABELS.get(name,name)}: PSNR {m['psnr']:.4f}dB, SSIM {m['ssim']:.4f}, pose reproduction max error {m['pose_reproduction_max_abs_error']:.2g}." for name,m in probe.items())
    text=f'''# 학생회관 추가 연구 및 직접 실행 보고서 · 2026-10-27

## 결론
{status}
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
{table}

보존 모델에는 시작 step5000도 포함된다. 따라서 step5000이 최고인 경우 추가 학습 자체가 나아졌다는 뜻이 아니다. 지표는 저장 PNG를 float64로 읽어 계산한 PSNR, Gaussian SSIM(sigma1.5, window11, population covariance)이다. 이전 화면 반올림 24.66dB와 같은 기준이며 재렌더링의 약0.00002dB 수치 차이는 유의미한 개선이 아니다. 채택 기준은 이전 대비 +0.05dB 이상, SSIM 하락0.002 이내다. 이는 통계적 유의성 시험이 아니라 작은 수치 변동에 따른 교체를 피하기 위한 실무 규칙이다.

## 오른쪽 사진 추가 진단
{probe_text}
오른쪽 5장은 외관 학습과 checkpoint 선택에 쓰지 않았다. 정답 사진에 pose나 exposure를 맞추지 않고, 기존 SfM 위치와 오른쪽 내부값으로 렌더링했다. 전체47카메라의 동일 정규화를 재현하고 실제 Dataset train pose와 오차1e-5 미만을 확인했다. 왼쪽 평균에 섞지 않고 별도 표기한다. SfM에는 전체 사진이 사용되어 외부 blind test는 아니다.

## 한계와 다음 실험의 우선순위
이번 추가 실험은 RGB 외관 최적화다. 건물 치수나 LiDAR 구조 정확도가 향상됐다고 주장하지 않는다. 관측되지 않은 바닥, 유리 반사, 가는 난간, 나무의 다른 시점에는 번짐이 남는다. 학습을 늘리거나 원본 해상도를 올리는 것이 항상 좋은 결과를 주지는 않았다.
다음 구조 중심 실험의 우선순위는 (1) SfM–LiDAR 좌표계/센서 시간·외부 보정 오차를 먼저 수치화하고, (2) 정렬 신뢰도가 높은 실제 LiDAR 깊이만 3D Gaussian RGB 학습에 함께 넣고, (3) 독립적으로 남긴 촬영 시점의 raw ray와 사진을 동시에 평가하는 것이다. 현재 깊이를 mesh로 만든 것과 Gaussian 학습에 직접 깊이 제약을 넣는 것은 구분해야 한다.
공식 gsplat의 현재 main에는 LiDAR sensor/rasterization 기능도 있지만 CUDA 기반이다. GPU 환경이 마련되면 같은 사진·카메라·검증 분할로 기본3DGS→antialiased/AbsGrad→depth supervision을 하나씩 비교하는 것이 후속 작업이다. 건물 평면·계단의 surface 자체가 목표면 PGSR을 별도 후보로 비교할 수 있다. 이 장비는 Apple M4이며 CUDA가 없어 이 공식 구현들을 직접 실행했다고 말하지 않는다. 새 클라우드 GPU 비용도 발생시키지 않았다.

## 보존·재현과 배포
원본195개와 이전 선택 모델4개 파일의 SHA-256을 재확인했다. 전체 checkpoint/PLY/splat과 모든 후보는 이 실험 폴더에 남겼다. 공개 ZIP에는 명령·로그·소스·native patch·환경·분할·색상 보정값·평가·비교 그림·입출력 hash를 넣는다. 큰 사진/HDF5/체크포인트는 중복 포함하지 않는다. .splat은 SH 시점별 색상을 단순화한 간편 뷰어용이며, PNG/영상은 전체 학습 모델로 렌더링했다.
브라우저 연결은 사용 가능한 브라우저가 없어 실패했다. 정적 링크·자산·JS와 공개 HTTP·다운로드 검증을 수행하며, 실제 UI 조작은 수행하지 않았다. 배포 영수증과 공개 HTTP 검증은 배포 후 로컬에 따로 저장한다.

## 실제 완료 실행 목록
{runlist}

## 조사한 논문·공식 구현과 판단
{references}
'''
    text=text.replace(str(ROOT),'/PROJECT').replace(str(Path.home()),'/USER_HOME');(O/'REPORT.md').write_text(text);(ROOT/'STUDENT_HALL_RESEARCH_20261027.md').write_text(text);(SITE/'hall-research-report.txt').write_text(text)
    rowshtml=''.join(f'<tr><td>{html.escape(x["label"])}</td><td>{x["psnr"]:.3f}</td><td>{x["ssim"]:.4f}</td><td>{x["step"]}</td><td>{x["gaussians"]:,}</td><td>'+(f'{x["best_after_training"]["psnr"]:.3f}' if x.get('best_after_training') else '—')+'</td></tr>' for x in [base]+r['candidates'])
    refshtml=''.join(f'<li><a href="{s["url"]}" target="_blank" rel="noopener">{html.escape(s["title"])} ↗</a></li>' for s in sources)
    figures=''.join(f'<article class="research-view"><h3>검증 시점 {frame}</h3><div class="research-grid">'+''.join(f'<figure><a href="assets/hall-research-{k}-{tag}.png" target="_blank" rel="noopener"><img src="assets/hall-research-{k}-{tag}.png" loading="lazy" alt="시점 {frame} {label}"></a><figcaption>{label}</figcaption></figure>' for tag,label in [('input','원본 사진'),('before','이전 선택 모델'),('candidate',LABELS[comparison['name']])])+'</div></article>' for k,frame in enumerate(VAL))
    ptext=''.join(f'<p>{html.escape(LABELS.get(n,n))}: PSNR <strong>{m["psnr"]:.3f}dB</strong> · SSIM {m["ssim"]:.4f}</p>' for n,m in probe.items())
    page=f'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>학생회관 · 추가 연구와 실험</title><meta name="description" content="원본 해상도·좌우 색상·Gaussian 재분할을 직접 비교한 학생회관 추가 실험. 논문 조사와 실패 기록까지 공개합니다."><link rel="stylesheet" href="style.css"><link rel="stylesheet" href="refinement.css"><link rel="stylesheet" href="hall-research.css"></head><body>
<header class="topbar"><a class="brand" href="index.html">▧ RECONSTRUCTION <span class="brand-light">LAB</span></a><nav aria-label="주요 메뉴"><a href="student-hall-20261027.html">이전 결과·3D 표면</a><a href="#trials">실험 비교</a><a href="#records">기록</a></nav></header><main>
<section class="hero"><div class="hero-copy"><p class="eyebrow"><span class="live-dot"></span> STUDENT UNION / FURTHER RESEARCH</p><h1>다음 방법을 찾고,<br><span>직접 비교했습니다.</span></h1><p class="hero-description">원본 해상도, 좌우 사진의 색상,<br>큰 Gaussian의 재분할까지.</p><p class="hero-footnote">논문과 공식 구현을 확인하고 같은 자료로 시험했습니다.<br>{'채택 기준을 넘은 새 모델을 공개합니다.' if adopt else '기준을 넘지 못한 후보는 기록하고 이전 모델을 유지했습니다.'}</p><a class="primary-link" href="#trials">실험 결과 보기 ↓</a></div><div class="hero-image"><img src="assets/hall-research-poster.jpg" alt="현재 선택 모델에서 그린 학생회관 계단과 입구" fetchpriority="high"><div class="photo-overlay"><span>STUDENT UNION BUILDING</span><span>현재 선택 모델</span></div></div></section>
<div class="summary"><div><span class="stat-label">현재 선택 PSNR ↑</span><strong>{selected['psnr']:.2f} dB</strong><span>같은 왼쪽 검증 사진 5장</span></div><div><span class="stat-label">현재 선택 SSIM ↑</span><strong>{selected['ssim']:.4f}</strong><span>사진 외관 일치도</span></div><div><span class="stat-label">비교 후보</span><strong>5가지</strong><span>재렌더링 1 + 추가 학습 4</span></div><div><span class="stat-label">추가 확인</span><strong>오른쪽 5장</strong><span>checkpoint 선택에 사용하지 않음</span></div></div>
<section class="section" id="trials"><div class="section-title"><div><p class="eyebrow">01 / CONTROLLED EXPERIMENTS</p><h2>{'새 후보를 선택했습니다.' if adopt else '더 학습해도,<br>항상 좋아지지는 않았습니다.'}</h2></div><p>{html.escape(status)}<br>물리적 LiDAR 표면은 이번 실험에서 변경하지 않았습니다.</p></div><div class="video-frame"><video controls muted playsinline loop preload="metadata" poster="assets/hall-research-poster.jpg"><source src="assets/hall-research-flythrough.mp4" type="video/mp4"></video><div class="video-caption"><span>CURRENT SELECTED MODEL</span><span>72 INTERPOLATED VIEWS</span></div></div><img class="research-chart" src="assets/hall-research-curves.png" alt="추가 학습 횟수에 따른 검증 화질. 이전 선택 모델의 점선과 후보별 변화를 함께 표시" loading="lazy"><div class="hall-table-wrap"><table><thead><tr><th>후보</th><th>보존 모델 PSNR ↑</th><th>SSIM ↑</th><th>선택 step</th><th>Gaussian</th><th>추가 학습 후 최고 PSNR</th></tr></thead><tbody>{rowshtml}</tbody></table></div><p class="hall-metric-note">step5000은 시작 모델입니다. 보존 모델의 step이5000이면 추가 학습보다 시작점이 좋았다는 뜻입니다. 채택 기준은 이전 대비 +0.05dB 이상, SSIM 하락0.002 이내입니다. 같은 검증 사진을 여러 번 사용했으므로 독립 시험이 아닙니다.</p></section>
<section class="section"><div class="section-title"><div><p class="eyebrow">02 / WHAT CHANGED</p><h2>원본에서 다시 준비하고,<br>조건을 나눠 비교했습니다.</h2></div></div><div class="method-steps"><article><span class="step-num">01</span><h3>실제 원본 해상도</h3><p>960px 사본을 확대하지 않고 좌우1440px 원본을 사용했습니다. 점 수 고정·학습률0.25배로 960/1440 후보를 각4,000회 추가 학습했습니다.</p></article><article><span class="step-num">02</span><h3>좌우 카메라 색상</h3><p>학습 사진의 같은 특징점 patch로 오른쪽 사진의 RGB gain/bias를 맞췄습니다. 검증 시점 양쪽 사진은 보정값 계산에서도 제외했습니다.</p></article><article><span class="step-num">03</span><h3>점 재분할과 처리 한도</h3><p>opacity reset을 끄고 큰 Gaussian을 표준 gradient 기준으로 다시 분할했습니다. 별도 렌더러의 한도2배 변경도 확인했지만, 단독 효과는 거의 없었습니다.</p></article></div><p class="hall-metric-note">고해상도 렌더링을 줄여 그리는 것은 Mip-Splatting 전체 구현이 아닙니다. 재분할은 AbsGS 구현이 아닙니다. 이번 학습에는 LiDAR 깊이 손실을 새로 넣지 않았습니다.</p></section>
<section class="section"><div class="section-title"><div><p class="eyebrow">03 / SAME VALIDATION VIEWS</p><h2>원본, 이전 모델,<br>추가 후보를 나란히.</h2></div><p>비교 후보: {html.escape(LABELS[comparison['name']])}<br>사진을 누르면 원래 크기로 열립니다.</p></div>{figures}<div class="reading-guide"><strong>다른 눈에서도 확인했습니다.</strong>{ptext}<p>오른쪽 진단 사진은 외관 학습과 checkpoint 선택에서 제외했습니다. SfM 위치 추정에는 포함됐으므로 외부 정답 시험은 아닙니다. 정답 사진에 pose·노출을 새로 맞추지 않았습니다.</p></div></section>
<section class="section method-section" id="records"><div class="section-title"><div><p class="eyebrow">04 / RESEARCH & RECORDS</p><h2>다음은,<br>깊이와 표면 제약입니다.</h2></div><p>현재 결과와 문헌을 함께 본 후속 우선순위입니다.<br>아래 공식 구현을 이번에 모두 실행했다는 뜻은 아닙니다.</p></div><div class="reading-guide"><strong>구조를 더 개선하기 위한 순서</strong><p>① SfM–LiDAR 정렬과 센서 보정 오차를 먼저 확인 → ② 신뢰할 수 있는 실제 LiDAR 깊이를 Gaussian 학습에 직접 추가 → ③ 별도로 남긴 사진과 LiDAR ray에서 함께 검증.</p><p>gsplat·PGSR·Gaussian-LIC2 공식 구현은 CUDA 기반입니다. 현재 Apple M4에서는 같은 코드를 바로 실행할 수 없어 이번에는 Metal에서 가능한 후보를 시험했습니다. 새로운 GPU 비용은 발생시키지 않았습니다.</p></div><ul class="research-references">{refshtml}</ul><div class="refine-downloads"><a class="primary-link" href="downloads/student_hall_research_20261027_records.zip" download>이번 연구·실험 기록 받기 (.zip) ↗</a><a class="text-link" href="hall-research-report.txt" target="_blank" rel="noopener">상세 보고서 읽기 ↗</a><a class="text-link" href="assets/hall-research-results.json" target="_blank" rel="noopener">전체 평가 수치 (.json) ↗</a><a class="text-link" href="downloads/student_hall_latest.splat" download>현재 선택 모델 (.splat) ↗</a></div><div class="limits"><h3>원본과 이전 결과를 보존했습니다.</h3><ul><li>공개 ZIP에는 명령·로그·코드·native patch·환경·분할·평가·비교 그림·해시가 있습니다. 큰 원본과 전체 모델·체크포인트는 프로젝트 실험 폴더에 보존했습니다.</li><li>오프라인 Metal 빌드 실패와 런타임 컴파일로 해결한 과정도 기록했습니다.</li><li>바닥·난간·나무·유리 주변에는 번짐이 남습니다. 외관 지표를 건물 치수 정확도로 해석하지 않습니다.</li><li>.splat은 시점별 SH 색상을 단순화한 간편 뷰어용입니다. 영상과 평가 사진은 전체 학습 모델로 렌더링했습니다.</li></ul></div></section></main><footer class="footer"><a class="brand" href="student-hall-20261027.html">▧ 이전 복원과 3D 표면 보기 ↗</a><span>STUDENT UNION · RESEARCH ROUND · 2026.10.27</span></footer></body></html>'''
    (SITE/'student-hall-research.html').write_text(page)
    (SITE/'hall-research.css').write_text('.research-chart{width:100%;max-width:1100px;display:block;margin:28px auto}.research-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.research-grid figure{margin:0}.research-grid img{width:100%;height:auto;display:block}.research-grid figcaption{font-size:13px;margin:10px 0 20px;color:#5d685f}.research-view{margin:30px 0}.research-references{line-height:1.8}.research-references a{color:inherit}.hall-table-wrap{overflow-x:auto}.hall-table-wrap table{width:100%;border-collapse:collapse;min-width:800px}.hall-table-wrap th,.hall-table-wrap td{text-align:left;padding:15px 12px;border-bottom:1px solid #d9e0d8;font-size:14px}.hall-metric-note{line-height:1.8;font-size:14px;color:#5d685f;margin-top:20px}.research-latest{padding:16px 5%;background:#e3ebe2;line-height:1.6}.research-latest a{color:#254932;font-weight:600}@media(max-width:720px){.research-grid{grid-template-columns:1fr}.research-grid figcaption{margin-bottom:16px}}')
    old=SITE/'student-hall-20261027.html';previous=old.read_text()
    if 'class="research-latest"' not in previous:old.write_text(previous.replace('<main>','<div class="research-latest">추가 연구·실험을 완료했습니다. <a href="student-hall-research.html">원본 해상도·색상·재분할 비교와 기록 보기 ↗</a></div><main>',1))
    site_snap=O/'site_snapshot';site_snap.mkdir(exist_ok=True)
    for name in ['student-hall-research.html','hall-research.css','style.css','refinement.css']:shutil.copyfile(SITE/name,site_snap/name)
    print('REPORT_AND_PAGE',status,flush=True)
def records():
    # Final pass after all recorded scientific runs finish, so no run is shown as running.
    assert all(read(p)['status']!='running' for p in (O/'runs').glob('*.json'))
    report();files=[]
    for pat in ['*.json','*.jsonl','*.md','*.patch','*.txt','*.png','*comparison.jpg','runs/*.json','logs/*.log','source_snapshot/*','site_snapshot/*','probe_inputs/plan.json','*/manifest.json','*/history.json','*/best_metrics.json','*/right_probe_metrics.json']:
        files.extend(p for p in O.glob(pat) if p.is_file() and p.name not in ['records_manifest.json','publication.json','public_http_checks.json'])
    # The full original native project is small after excluding datasets/builds.
    files.extend(p for p in (O/'native_source').rglob('*') if p.is_file() and p.suffix not in ['.so','.pyc'] and '__pycache__' not in p.parts)
    dest=SITE/'downloads/student_hall_research_20261027_records.zip'
    with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in sorted(set(files)):
            key=str(p.relative_to(O))
            if p.suffix in ['.png','.jpg']:z.write(p,key)
            else:z.writestr(key,p.read_text(errors='replace').replace(str(ROOT),'/PROJECT').replace(str(Path.home()),'/USER_HOME'))
    with zipfile.ZipFile(dest) as z:assert z.testzip() is None;entries=len(z.namelist())
    dump(O/'records_manifest.json',dict(path=str(dest),bytes=dest.stat().st_size,sha256=sha(dest),entries=entries,contains_large_originals_and_checkpoints=False));print('RECORDS',entries,dest.stat().st_size,sha(dest),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['evaluate','assets','provenance','report','records']);a=p.parse_args();globals()[a.stage]()
