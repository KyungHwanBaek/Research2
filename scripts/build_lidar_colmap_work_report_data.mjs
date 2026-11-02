import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import crypto from 'node:crypto';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const output = path.join(root, 'outputs/lidar_colmap_01a18370_20261029');
await fs.mkdir(path.join(output, '.build'), {recursive:true});
const origin = 'https://stereo-reconstruction-lab-kyunghwan.sillycod2.chatgpt.site';
const read = async p => JSON.parse(await fs.readFile(path.join(root,p),'utf8'));
const r26 = await read('experiments/student_hall_refinement_20261026/results.json');
const r27 = await read('experiments/student_hall_refinement_20261027/results.json');
const research = await read('experiments/student_hall_research_20261027/results.json');
const initial = await read('experiments/student_hall_lidar/evaluation.json');
const pages = {initial:'student-hall.html',refine:'student-hall-next.html',follow:'student-hall-20261027.html',research:'student-hall-research.html'};
const work = [];
function add(date,stage,category,title,purpose,method,result,decision,limits,artifact,source,page) {
  work.push([`W${String(work.length+1).padStart(2,'0')}`,date,stage,category,title,purpose,method,result,decision,limits,artifact,source,page?`${origin}/${page}`:'']);
}
add('2026-10-21','최초 결합','조사','LiDAR–COLMAP 결합 논문·기법 조사','학생회관에서 기존 사진 복원과 LiDAR를 함께 쓰는 방법 선정','Colmap-PCD 문제 설정, KISS-ICP, Open3D 정합, LI-GS 계열 검토','정합·포즈 그래프·표면과 외관 비교 파이프라인으로 진행','설계에 반영','Colmap-PCD·LI-GS 공식 전체 구현 재현은 아님','학생회관 결합 보고서','STUDENT_HALL_LIDAR_REPORT.md',pages.initial);
add('2026-10-21','최초 결합','입력','48개 촬영 시점 자료 점검','친구의 완성 지도 없이 사용 가능한 실제 자료 확인','01-26-15-41-24의 RGB·원시 LiDAR·보정값·storage.hdf5 사용','원시 LiDAR 48개와 촬영 시점 48개 확인','사용','친구가 만든 누적 지도·궤적은 미제공. 데이터 식별자의 01-26은 작업 날짜 아님','frames/; source.json','experiments/student_hall_lidar/source.json',pages.initial);
add('2026-10-21','최초 결합','좌표 정렬','COLMAP 두 컴포넌트를 미터 단위로 보정','사진 복원의 임의 크기를 LiDAR와 비교 가능한 크기로 조정','특징점 관측 깊이/LiDAR 깊이 비율 중앙값, 공통 카메라 배치','중복 제외 카메라 40개·특징점 5,183개','비교 기준선','임시 route-stitched 경로 합본은 정밀 정합 입력으로 쓰지 않음','미터 보정 COLMAP','STUDENT_HALL_LIDAR_REPORT.md',pages.initial);
add('2026-10-21','최초 결합','정합','KISS-ICP 단독 추적 대조군','LiDAR만으로 연결할 때의 수준 확인','KISS-ICP 1.3.0로 48스캔 추적','25cm 겹침 70.96%, 대응 접평면 중앙값 2.866cm','미채택','스캔 내 타임스탬프가 없어 motion deskew 없음. 스캔 사이 수 초 간격','kiss 추적 결과','experiments/student_hall_lidar/evaluation.json',pages.initial);
add('2026-10-21','최초 결합','정합','영상·LiDAR 초기값과 robust ICP','회전·추적 단절을 줄이며 전체 시점 연결','영상 특징+LiDAR PnP, COLMAP 상대 위치, KISS 초기값을 다중 해상도 point-to-plane ICP로 비교','순차 ICP 겹침 77.79%, clipped plane mean 8.119cm','포즈 그래프 입력','대응점 오차는 실측 정답 오차가 아님','icp_odometry 후보','experiments/student_hall_lidar/evaluation.json',pages.initial);
add('2026-10-21','최초 결합','정합','FPFH/RANSAC 전역 회전 복구','어려운 회전 시점의 정합 개선','FPFH 특징·RANSAC 전역 후보 후 ICP 검증','17→18 시점 정합 겹침 약33.5→64.2%','사용','해당 시점 정합 fitness이며 전체 반환점 평가와 다른 지표','rescue 정합 기록','STUDENT_HALL_LIDAR_REPORT.md',pages.initial);
add('2026-10-21','최초 결합','정합','포즈 그래프 최적화 및 구조 선택','여러 시점 간 LiDAR 정합 관계를 함께 만족','인접·건너뛴 시점의 검증된 관계를 공동 최적화','48시점 연결, 겹침 77.81%, 대응 중앙값 2.690cm','구조에 채택','동일 37쌍·홀수 반환점 668,519개 비교. 측량 정확도 아님','posegraph_poses.npz','experiments/student_hall_lidar/evaluation.json',pages.initial);
add('2026-10-21','최초 결합','공동 최적화','카메라·특징점·LiDAR 평면 공동 보정','영상과 LiDAR를 동시에 제약하는 후보 검증','4,812점·23,283 영상 관측·2,327 접평면 제약 사용','남긴 관측 재투영 중앙값 4.027→0.524px, 스캔 겹침은74.69%로 하락','구조에는 미채택','남긴 3,327관측은 이번 보정에서만 제외. 원래 SfM에는 사용','joint_poses.npz; joint_report.json','experiments/student_hall_lidar/joint_report.json',pages.initial);
add('2026-10-21','최초 결합','표면','TSDF·Poisson 표면 비교와 거짓 면 제거','점들을 연결해 계단·입구 표면 생성','TSDF voxel 6/10/15cm, Poisson depth10/11 비교. 센서3m 이내 반환점·관측에서25cm 넘게 먼 정점 제거','선택 Poisson 원본2,120,479삼각형, 전체 LiDAR1,648,179점·컬러232,136점','Poisson 채택','검증용6스캔 제외 후 최종은 전체 스캔 사용. 빈 곳·잡음 잔존','fused/student_hall_poisson.ply; lidar_full_map.ply','experiments/student_hall_lidar/fused/poisson_validation.json',pages.initial);
add('2026-10-21','최초 결합','외관','COLMAP·공동 보정·LiDAR 점 초기화 3DGS 비교','사진 사이 시점의 외관 모델 선택','21장 학습·왼쪽5장 검증, RGB 손실, 12,000/30,000회 비교','COLMAP12k PSNR23.281dB·native SSIM0.7181. LiDAR초기화15.778dB','COLMAP12k 채택','최종3DGS에 LiDAR 깊이 손실은 없음. 이후 공통 PNG평가 SSIM과 구분','gs_colmap_12000/','STUDENT_HALL_LIDAR_REPORT.md',pages.initial);
add('2026-10-21','최초 결합','공유·검증','학생회관 웹 결과 공개','친구가 파일 설치 없이 결과 확인','축소 메시·LiDAR 점군·원본 사진·렌더링 영상·보고서 제공, 버퍼·JS·HTTP 검사','공개 학생회관 페이지 배포','공개','당시 실제 브라우저 마우스/터치 검사는 미실시','student-hall.html; public_deployment.json','experiments/student_hall_lidar/public_deployment.json',pages.initial);
add('2026-10-22','설명·전달','문서','복원 과정과 결과 Word 작성','어떤 자료·방법·결과인지 전달','입력, 좌표 정렬, LiDAR 누적, 표면과 외관 관계·한계 문서화','설명 DOCX 보관, 구조·접근성 점검','완료','10월21일 당시 결과 설명이며 후속 최고 모델 설명은 아님','docs/학생회관_3D복원_과정과결과_정리.docx','docs/student_hall_explanation/structural_audit.json','download-report.html');
add('2026-10-22','설명·전달','공유·검증','Word 다운로드 경로 수정','로컬 링크 클릭 시 다운로드 문제 해결','기존 공개 사이트에 다운로드 페이지·DOCX 제공, HTTP·SHA-256·ZIP 검사','HTTP200, 4,114,590바이트, 원본과 해시 일치','완료','다운로드 해결 작업이며 새 복원 성능 실험은 아님','downloads/student_hall_3d_reconstruction_report.docx','docs/student_hall_explanation/web_download_verification.json','download-report.html');
add('2026-10-26','1차 개선','보정','스테레오 외부 보정 점검','좌우 사진 정렬 개선','기준 길이0.132133m 유지, LiDAR 깊이 특징의 우측 재투영을 추가한 작은 보정','제외한 좌우쌍 수직 오차 중앙값 약0.94→0.55px','후속 시험 사용','이동 방향이 반전된 무제약 essential 보정은 거부. 깊이 정확도 입증은 아님','stereo calibration 기록','STUDENT_HALL_REFINEMENT_REPORT.md',pages.refine);
add('2026-10-26','1차 개선','깊이','RGB/NIR 스테레오 깊이 생성·보정','LiDAR가 빈 부분을 채울 가능성 확인','48쌍 SGBM·좌우 일관성·질감·거리 필터, LiDAR anchor80/20 공간 분할로 시차 scale/bias 검증','raw_rgb_depth/raw_nir_depth와 보정 후보 보존','실험 수행','NIR를 정렬 검증 없는 독립 보완 깊이로 합치지 않음','스테레오 깊이 및 보정 JSON','STUDENT_HALL_REFINEMENT_REPORT.md',pages.refine);
add('2026-10-26','1차 개선','표면','다중 시점 스테레오+LiDAR hybrid 표면','LiDAR 빈 영역을 신뢰 가능한 RGB 깊이로 보완','다른 두 시점 이상 일치·자유 공간·동일 시점 LiDAR 모순 검사, 검증6시점 전부 제외','학생회관 검증23.1998cm, 기존20.3786cm보다 나쁨','미채택','누락 포함1m 상한 페널티 지표. 전체6시점 평균과 학생회관4시점 평균 구분','hybrid_v0.080_d11','experiments/student_hall_refinement_20261026/results.json',pages.refine);
add('2026-10-26','1차 개선','표면','세밀한 Poisson과 기존 함수 재실행 대조','더 촘촘한 표면이 좋은지 확인','6cm/depth11·8cm/depth12, 작은 조각·먼 정점 제거, baseline 함수 그대로 재현','학생회관23.4248·22.8788cm. baseline20.3786cm 재현','미채택','삼각형 증가가 구조 정확도 향상을 뜻하지 않음','surface_candidates; baseline control','experiments/student_hall_refinement_20261026/results.json',pages.refine);
add('2026-10-26','1차 개선','정합','전체 LiDAR 스캔 자세 재보정','국소 ICP를 넘어 전체 스캔 일관성 개선','약74,700 대칭 평면 제약을 반복 갱신, 짝수로 fitting·홀수로 비교','clipped plane mean8.1439→8.1305cm, 표면20.3786→20.5183cm','미채택','스캔 지표 소폭 개선이 학생회관 표면 개선으로 이어지지 않음','global_poses.npz; global_pose_evaluation.json','experiments/student_hall_refinement_20261026/results.json',pages.refine);
add('2026-10-26','1차 개선','SfM','좌우52장 원본 해상도 COLMAP','학생회관 관측과 특징점 확대','프레임20~45 좌우1440×1080, SIFT최대11,000, exhaustive/guided matching, 고정 내부값 BA','52/52장 한 모델·32,820점·177,306관측·평균0.8758px','외관 학습에 사용','이전 주모델3,518점 대비9.3배는 정확도 배율 아님','stereo COLMAP 모델','experiments/student_hall_refinement_20261026/results.json',pages.refine);
add('2026-10-26','1차 개선','외관','좌우42장 3DGS 및 720/960px 추가 학습','같은 검증 사진 외관 개선','검증5시점의 양쪽 사진 제외, 16k학습+각6k추가 학습, 공통720×540 PNG 재평가','gs3d_fine_960 PSNR24.1397dB·SSIM0.7736·Gaussian436,661','당시 외관 채택','선택checkpoint12k. 학습 관측·SfM·해상도 함께 바뀜. 깊이 손실 없음','gs3d_fine_960/','experiments/student_hall_refinement_20261026/results.json',pages.refine);
add('2026-10-26','1차 개선','외관·깊이','Metal 2DGS RGB/depth-loss 대조','표면형 Gaussian과 깊이 제약 가능성 검증','같은 LiDAR surfel 초기화에서RGB-only/depth-loss각4k, 기존3DGS warm start도시험','공통평가16.0807/15.1073dB, warm22.9889dB','미채택','보고서 native 근삿값과 공통 PNG값 구분. 완전한 alpha 미분 깊이 최적화 아님','gs2d_* 후보','experiments/student_hall_refinement_20261026/results.json',pages.refine);
add('2026-10-26','1차 개선','직접 융합','새 COLMAP–LiDAR similarity 정렬','확대된 사진 점을 LiDAR와 합칠 가능성 확인','깊이 anchor·카메라로 robust similarity, landmark20% 제외. 3시점이상 관측·20cm지지 특징만 추가','anchor중앙22.7293cm·P90 69.1204cm, 새지지점3,502개','정밀 구조로 미채택','전역 similarity로 시점별 변형이 모두 해결되지 않음','expanded_colmap_metric; supported_fusion','experiments/student_hall_refinement_20261026/results.json',pages.refine);
add('2026-10-26','1차 개선','표면 색상','가시성 기반 실제 사진 색상 투영','회색으로 남는 관측 가능 표면 개선','정점을 사진에 투영, ray casting 차폐 검사, 각도·거리·가장자리 가중 RGB조합','사진색정점120,362→397,611 /1,110,732 (10.84→35.80%)','채택','원래 정점좌표·삼각형 동일 확인. 색상 개선이며 새 구조 측정 아님','baseline_recolored; assets/s16-mesh.ply','experiments/student_hall_refinement_20261026/results.json',pages.refine);
add('2026-10-26','1차 개선','기록·공유','실행·오류·해시 기록과 개선 페이지 공개','재현 및 이전 결과 보존','41실행(39성공2실패), OpenCV3×1형상·OpenMP충돌 수정, 원본195파일·소스·환경·로그 보존','기존주요9파일SHA동일, 공개ZIP·사이트버전4','완료','ZIP은 큰 원본·전체checkpoint 미포함. 개인 홈 경로 치환','runs/; records.zip; publication.json','experiments/student_hall_refinement_20261026/publication.json',pages.refine);
add('2026-10-27','2차 개선','SfM','96장 전체 촬영 SfM 확장','학생회관 앞뒤 경로까지 관측 확대','DB backup으로52장ID·대응 보존,44장 추가,4,560쌍 matching·전역 BA','88/96장·56,257점·278,436관측·평균0.8065px','확장 후보 보존','18·19·46·47 좌우8장 미연결. 늘어난 점에 주변 경로 포함','full_sfm/','experiments/student_hall_refinement_20261027/results.json',pages.follow);
add('2026-10-27','2차 개선','외관','주변14장 추가 context 학습','경로 확장이 학생회관 화질도 높이는지 검증','100개이상 특징 공유하는7시점 좌우14장 추가,42→56장 학습','PSNR23.9050dB, 최고 모델보다 낮음','미채택','사진 수 증가가 화질 향상을 보장하지 않음','gs_context_720_12000/','experiments/student_hall_refinement_20261027/results.json',pages.follow);
add('2026-10-27','2차 개선','카메라 보정','focal·principal point bundle adjustment','카메라 내부 보정을 다시 맞춤','공유focal만 조정·focal+principal 조정, 52장 관측 동일','평균 재투영0.8758→0.6911→0.6471px','principal 외관 후보 사용','내부값·자세·점 동시 보정. 절대 건물 치수 정확도 지표 아님','ba_focal; ba_principal','experiments/student_hall_refinement_20261027/results.json',pages.follow);
add('2026-10-27','2차 개선','대조 실험','fixed/focal/principal 동일 조건 비교','보정 후보 간 비교 조건 정비','42학습·5검증·seed42·12k예산, 초기점28,892개·초기 색도 학습사진만 사용','고정24.0414/focal24.0873/principal24.1647dB','principal 후속 사용','이전32,820점 초기화 모델과는 조건 일부 다름. SfM은 전체 사진 사용','gs_fixed/focal/principal_720_12000','experiments/student_hall_refinement_20261027/results.json',pages.follow);
add('2026-10-27','2차 개선','외관','좋은 중간 모델 보존·960px 고정 점 추가 학습','opacity reset·무분별한 점 증가 이후 화질 저하 방지','principal step3000에서 분할·제거·reset 중단,Gaussian파라미터는 학습,추가6k예산 중 checkpoint선택','freeze_principal_960 step5000,24.6562dB·0.7946·211,669Gaussian','현재 최종 외관','선택step5000은기초3000+추가2000. 예산 끝모델이 아님','freeze_principal_960/; student_hall_latest.splat','experiments/student_hall_refinement_20261027/results.json',pages.follow);
add('2026-10-27','2차 개선','표면','물리 LiDAR 원점 360° TSDF','정면 카메라 깊이만 쓰던 TSDF 범위 보완','스캔당100°/512px 가상카메라6면, voxel6/10/15cm,검증6스캔 제외','학생회관 보간지표36.6659/40.3184/44.8037cm','미채택','6면은 같은 스캔의 투영이며 독립6관측 아님. 누락 벌점 큼','cube_tsdf 후보','experiments/student_hall_refinement_20261027/results.json',pages.follow);
add('2026-10-27','2차 개선','검증','보간하지 않은 실제 LiDAR ray 진단','표면이 실측 반환 방향·거리에 맞는지 추가 확인','물리원점 raw반환점4개마다1개,3~35m,1m상한·누락페널티,360와FOV 별도','기본표면 학생회관FOV raw24.0718cm','평가 추가','보간 깊이20.3786cm와 거리·표본이 달라 절대값 직접 비교 금지','baseline_raw; raw_rays JSON','experiments/student_hall_refinement_20261027/results.json',pages.follow);
add('2026-10-27','2차 개선','표면','LiDAR 국소 평면 MLS 보정 후보','기존 표면을 가까운 실측점에 제한적으로 맞춤','짝수반환점8cm voxel·PCA지지평면,반경25/35cm·이동상한5/8cm,방향반전정점복귀','안전후보 보간20.2174cm·raw23.7218cm,이동중앙6.47mm','후보만 공개','4시점 소폭 개선. 기본 구조 유지. 면/구멍 새 생성 없음','mls_r0.35_l0.08_safe; assets/s18-mesh.ply','experiments/student_hall_refinement_20261027/results.json',pages.follow);
add('2026-10-27','2차 개선','기록·공유','전후 비교·모델·실험 기록 공개','현재 선택 결과와 미채택 결과 전달','33실행 성공, 소스·환경·hash·학습이력·동영상·3D표면·기록ZIP 공개','사이트버전5,72개 보간시점 영상,간편.splat','완료','.splat은시점별SH색 단순화,평가PNG/영상은전체학습모델','publication.json; records.zip','experiments/student_hall_refinement_20261027/publication.json',pages.follow);
add('2026-10-27','추가 연구','조사','3DGS·Mip-Splatting·AbsGS·gsplat·PGSR·Gaussian-LIC2 조사','남은 번짐·aliasing·깊이 문제의 후속 방법 선정','공식코드·논문을 검토하고 Metal에서 가능한 항목과 CUDA필요 항목 구분','처리한도·원본해상도·학습색보정·표준재분할을 직접시험','조사 완료','Mip/AbsGS/PGSR/LIC2 전체재현은 아님. 클라우드GPU 지출 없음','research_sources.json','STUDENT_HALL_RESEARCH_20261027.md',pages.research);
add('2026-10-27','추가 연구','렌더러','타일 한도2048→4096·격리 native 빌드','기존 경고가 검증 화질을 제한하는지 진단','설치본 유지한소스복제,MSL runtime compile,overflow·학습률API 추가. xcrun실패기록보존','검증5장overflow0,재렌더링 약+0.0002dB','실질 개선 미채택','한도는무제한아님. 컴파일경로도변경. 모든시점안전보장아님','native_changes.patch; native_package/','experiments/student_hall_research_20261027/results.json',pages.research);
add('2026-10-27','추가 연구','외관','실제1440원본·저학습률 추가 학습','실제 원본 세부 정보의 효과 확인','960대조와1440원본 각4k,6개파라미터군학습률0.25배,500회검증','추가학습최고960:24.6015/1440:24.5922dB','미채택','960후보폴더최고24.6564는시작모델포함. 새학습성능과구분','native960_low_lr; native1440_low_lr','experiments/student_hall_research_20261027/results.json',pages.research);
add('2026-10-27','추가 연구','색상·외관','학습 사진만의 좌우 RGB affine 보정','좌우 노출·색상 차이를 줄이면 외관이 좋아지는지 시험','SfMtrack3×3patch중앙값,오른쪽gain0.7~1.4/bias±0.1,검증양쪽사진제외','추가학습최고24.5029dB','미채택','사진전처리이며공식3DGS의학습중exposure최적화와다름','stereo_color_fits.json; native1440_stereo_color','experiments/student_hall_research_20261027/results.json',pages.research);
add('2026-10-27','추가 연구','외관','고해상도 표준 Gaussian 재분할','큰 Gaussian이 세부 표현을 덮는지 검증','opacity reset없이500회마다표준signed-gradient densification,4k예산','추가학습최고24.2618dB,최종242,988Gaussian','미채택','AbsGS의absolute-gradient구현은아님. 점수증가없음','native1440_progressive_split/','experiments/student_hall_research_20261027/results.json',pages.research);
add('2026-10-27','추가 연구','검증','선택에 쓰지 않은 오른쪽5장 진단','왼쪽반복검증과분리한추가진단','정답사진에pose/exposure재조정없이기존우측내부값·SfM자세로렌더링,pose재현확인','기존24.3627dB/0.7968,1440후보24.3253dB/0.7969','기존 선택 유지','SfM에는전체사진사용. 외부blindtest아님. 왼쪽평균과혼합안함','right_eye_diagnostics','experiments/student_hall_research_20261027/results.json',pages.research);
add('2026-10-27','추가 연구','선택·공유','최종 모델 유지·실패 포함 기록 공개','수치 변동만으로 최고 모델 교체 방지','PSNR+0.05dB이상·SSIM하락0.002이내 규칙,19실행18성공1실패,해시·ZIP·HTTP검사','추가후보미채택,24.6562dB 모델 유지. 사이트버전6','현재 상태 확정','최종물리LiDAR표면이번라운드변경없음','student-hall-research.html; records.zip','experiments/student_hall_research_20261027/publication.json',pages.research);
add('2026-10-27','날짜별 정리','문서','프로젝트 날짜별 작업 보고서 작성','무엇을 언제 했는지 기존기록으로 재구성','실행JSON·보고서·사이트Git이력대조,10/5~10/27전체연혁정리','기존 날짜별보고서·evidence_index 보관','완료','새복원실험이아님. 이번엑셀은10/21결합요청이후로범위축소','docs/3d_reconstruction_chronology_20261027.md','docs/chronology_20261027/evidence_index.json','');

const publicLinks=[
['추천·현재 결과','10월27일 선택 결과·영상·3D표면','/student-hall-20261027.html','현재 최고 외관·기본LiDAR표면·국소보정후보','페이지','친구에게 먼저 보낼 링크'],
['최초 결합','10월21일 LiDAR+COLMAP 결과','/student-hall.html','최초 결합 방법·점군·표면·비교','페이지','후속개선 전 결과'],
['1차 개선','10월26일 개선 결과','/student-hall-next.html','좌우52장·외관·색상 개선·미채택 후보','페이지','24.14dB 당시 결과'],
['마지막 연구','10월27일 추가 연구와 실험','/student-hall-research.html','원본해상도·색상·재분할·렌더러한도 비교','페이지','추가후보미채택·기존최고유지'],
['최초 설명','복원 설명 Word 다운로드 페이지','/download-report.html','10월21일 결과 설명문서 다운로드','페이지','10월22일 작성, 최신결과 설명은 아님'],
['최초 설명','복원 과정·결과 Word','/downloads/student_hall_3d_reconstruction_report.docx','4.1MB DOCX','DOCX','10월21일결과 기준'],
['모델','현재 간편 Gaussian 모델','/downloads/student_hall_latest.splat','211,669 Gaussian 변환본','SPLAT','SH시점별색 단순화. 평가PNG·영상과 표현력동일하지않음'],
['모델','기본 선택 표면·사진 색상','/assets/s16-mesh.ply','웹표시용 약18만삼각형 PLY','PLY','로컬약212만삼각형 전체모델과구분'],
['모델','국소 평면 보정 안전 후보','/assets/s18-mesh.ply','MLS비교후보·웹표시용 PLY','PLY','기본구조로채택하지않음'],
['기록','10월26일 실험 기록','/downloads/student_hall_refinement_records.zip','명령·로그·코드·평가·해시','ZIP','원본사진/HDF5·전체checkpoint는미포함'],
['기록','10월27일 1차 실험 기록','/downloads/student_hall_followup_20261027_records.zip','카메라보정·고정점학습·표면후보 기록','ZIP','원본사진/HDF5·전체checkpoint는미포함'],
['기록','10월27일 추가 연구 기록','/downloads/student_hall_research_20261027_records.zip','조사·실행·native patch·실패 기록','ZIP','원본사진/HDF5·전체checkpoint는미포함'],
['상세 보고서','최초 LiDAR 결합 보고서','/student-hall-report.md','10월21일 작업 설명','문서','Markdown'],
['상세 보고서','10월26일 개선 보고서','/refinement-report.txt','수행방법·한계·실행목록','문서','텍스트'],
['상세 보고서','10월27일 1차 보고서','/hall27-report.txt','선택모델과구조후보 설명','문서','텍스트'],
['상세 보고서','마지막 추가 연구 보고서','/hall-research-report.txt','새후보미채택 이유·논문조사·한계','문서','텍스트']
];
const checkedAt=new Date().toLocaleString('sv-SE',{timeZone:'Asia/Seoul'})+' KST';
const links=await Promise.all(publicLinks.map(async(row)=>{
  const url=origin+row[2];
  try {const response=await fetch(url,{method:'HEAD',signal:AbortSignal.timeout(20000)});return [row[0],row[1],url,row[3],row[4],response.status===200?'공유 가능':`확인 필요 (${response.status})`,response.status,checkedAt,row[5]];}
  catch(e){return [row[0],row[1],url,row[3],row[4],'접속 확인 실패','',checkedAt,e.message];}
}));

const roundDefinitions=[['2026-10-26','1차 개선','student_hall_refinement_20261026'],['2026-10-27','2차 개선','student_hall_refinement_20261027'],['2026-10-27','추가 연구','student_hall_research_20261027']];
const executions=[];
const runMeaning={stereo_calibration:'스테레오보정 최초실행·이동벡터형상오류',stereo_calibration_columnT:'이동벡터3×1형상수정재실행',stereo_calibration_metric:'물리기준길이 유지한보정',stereo_depth_all:'48시점 RGB/NIR스테레오깊이 생성',stereo_bias:'LiDAR anchor 시차scale/bias 검증',stereo_multiview:'여러시점일치·자유공간검사',stereo_multiview_corrected:'시차보정후여러시점검사',global_multiscan_refinement:'전체스캔평면제약자세보정',global_surface_validation:'전체자세보정표면검증',baseline_surface_control:'원래표면함수대조군',surface_visibility_color:'표면가시성기반사진색상',stereo_sfm_lidar_alignment:'새SfM·LiDAR similarity정렬',fine_720:'Gaussian수고정720px추가학습',fine_960:'Gaussian수고정960px추가학습',freeze_principal_960:'최고중간모델고정점960px추가학습',ba_focal:'초점거리공유내부값BA',ba_principal:'초점거리·주점공동BA',native_isolated_build:'xcrun metal부재로첫native빌드실패',native_runtime_build:'MSL런타임컴파일로빌드성공',native1440_low_lr:'실제1440원본저학습률추가학습',native960_low_lr:'960대조군저학습률추가학습',native1440_stereo_color:'학습사진만의좌우색보정추가학습',native1440_progressive_split:'원본해상도표준재분할추가학습',wide_tile_render_check:'타일한도4096재렌더링·overflow진단'};
function meaning(n){
 if(runMeaning[n])return runMeaning[n];
 if(n.startsWith('stereo_sfm_')||n.startsWith('full_sfm_'))return 'COLMAP '+n.split('_').at(-1)+' 단계';
 if(n.startsWith('cube_tsdf_'))return '360도TSDF voxel '+n.split('_').at(-1)+' 후보';
 if(n.startsWith('gs_'))return '외관대조군 '+n;
 if(n.startsWith('mls_'))return '국소평면표면보정·방향검사';
 if(n.startsWith('2dgs_'))return 'Metal2DGS RGB/깊이/warm start 비교';
 if(n.startsWith('right_probe_'))return '선택에쓰지않은오른쪽사진진단';
 if(/provenance|record|manifest/.test(n))return '소스·명령·원본/산출물해시기록';
 if(/report|assets|evidence|public/.test(n))return '보고서·공개비교자산작성';
 if(/check|validation|qa|verify/.test(n))return '수치·파일·공개자산검증';
 if(/env|dependencies|fetch|smoke/.test(n))return '실행환경·의존성준비/작동검사';
 return '전처리·복원·검증 지원 실행 (명령 참조)';
}
for(const [date,stage,folder] of roundDefinitions){
 for(const filename of await fs.readdir(path.join(root,'experiments',folder,'runs'))){
  if(!filename.endsWith('.json'))continue;
  const p=`experiments/${folder}/runs/${filename}`,r=await read(p);
  const command=Array.isArray(r.command)?r.command.join(' '):String(r.command??'');
  executions.push([date,stage,r.name,meaning(r.name),r.start??'',r.end??'',r.status==='succeeded'?'성공':r.status==='failed'?'실패':r.status,r.exit_code??'',r.elapsed_s??'',command.replaceAll(root,'/PROJECT').replaceAll('/Users/baekkyunghwan','/USER_HOME'),p]);
 }
}
executions.sort((a,b)=>a[4].localeCompare(b[4]));

const appearances=[];
function app(date,stage,name,metric,gaussians,step,state,note,source){appearances.push([date,stage,name,metric.psnr,metric.ssim,gaussians??'',step??'',state,note,source]);}
for(const c of r26.appearance_candidates)app(c.name==='previous_colmap_12000'?'2026-10-21':'2026-10-26','공통 PNG 평가',c.name,c,c.training?.gaussians??(c.name==='previous_colmap_12000'?318953:null),c.training?.step,c.name===r26.selected_appearance.name?'10/26 채택':'비교 후보','공통720×540 PNG·같은왼쪽검증5장. 2DGS native 근삿값과구분','experiments/student_hall_refinement_20261026/results.json');
for(const c of r27.candidates)app('2026-10-27','1차 후속',c.name,c,c.gaussians,c.checkpoint,c.name===r27.selected.name?'현재 채택':'미채택','fixed/focal/principal은같은학습42장·초기28,892점·seed42','experiments/student_hall_refinement_20261027/results.json');
for(const c of research.candidates){
 if(c.name==='wide_render'){app('2026-10-27','추가 연구·재렌더링',c.name,c,c.gaussians,c.step,'기존 모델 유지','학습0회·타일한도확장 재렌더링. 약+0.0002dB를개선으로미채택','experiments/student_hall_research_20261027/results.json');continue;}
 const t=c.best_after_training;
 app('2026-10-27','추가 연구·실제 추가학습',c.name,t,t.gaussians,t.step,'미채택',`시작모델포함 폴더최고 ${c.psnr.toFixed(4)}dB (step${c.step}). 이 행은추가학습후 최고치`,'experiments/student_hall_research_20261027/results.json');
}
const nativeInitial=[];
for(const [name,folder] of [['colmap12k','gs_colmap_12000'],['colmap30k','gs_colmap_30000'],['lidar12k','gs_lidar_12000'],['sparse12k','gs_sparse_12000']]){
 const p=`experiments/student_hall_lidar/${folder}/metrics.json`;try{nativeInitial.push({name,source:p,metrics:await read(p)});}catch{}
}
const perView=[];
for(const [label,v] of [['10/21 원본·공통재평가',r26.baseline_appearance],['10/26 선택',r26.selected_appearance],['10/27 현재 선택',r27.selected]])for(const view of v.views)perView.push([label,view.frame,view.psnr,view.ssim,'왼쪽·공통720×540·검증사진은외관학습에서제외']);

const surface=[];
const baselineRaw=r27.baseline_raw.hall_raw_ray_mae_cm;
surface.push(['2026-10-26','기본 Poisson 재현',r26.baseline_surface.student_hall_mae_cm,baselineRaw,'기본 구조','학생회관4시점 평균. 보간과raw는다른거리정의','experiments/student_hall_refinement_20261026/results.json']);
for(const c of r26.surface_candidates)surface.push(['2026-10-26',c.name,c.student_hall_mae_cm,'',c.name==='baseline_baseline_parameters'?'대조군':'미채택','보간깊이·1m상한·누락벌점','experiments/student_hall_refinement_20261026/results.json']);
for(const c of r27.surfaces)surface.push(['2026-10-27',c.name??`cube_tsdf_${c.voxel_m}`,c.student_hall_mae_cm,c.raw_rays?.hall_raw_ray_mae_cm??'',c.name==='mls_r0.35_l0.08_safe'?'후보 공개·기본 미교체':'미채택',c.name?.includes('safe')?'면방향반전정점복귀한안전후보':'표면생성에서검증6스캔제외','experiments/student_hall_refinement_20261027/results.json']);

const sources=[];
const sourcePaths=['STUDENT_HALL_LIDAR_REPORT.md','STUDENT_HALL_REFINEMENT_REPORT.md','STUDENT_HALL_FOLLOWUP_20261027.md','STUDENT_HALL_RESEARCH_20261027.md','docs/3d_reconstruction_chronology_20261027.md','docs/chronology_20261027/evidence_index.json','docs/student_hall_explanation/web_download_verification.json','experiments/student_hall_lidar/evaluation.json','experiments/student_hall_lidar/joint_report.json','experiments/student_hall_lidar/fused/poisson_validation.json','experiments/student_hall_refinement_20261026/results.json','experiments/student_hall_refinement_20261027/results.json','experiments/student_hall_research_20261027/results.json'];
for(const p of sourcePaths){const bytes=await fs.readFile(path.join(root,p));sources.push([p,crypto.createHash('sha256').update(bytes).digest('hex'),bytes.length,'프로젝트 상대경로. 원본 근거 보존']);}
const payload={createdKst:checkedAt,scopeStartKst:'2026-10-21 18:52:26 KST',scopeEnd:'2026-10-27 (가장 최근 기존 작업)',origin,work,links,executions,appearances,perView,surface,scanConsistency:initial.metrics,nativeInitial,sources,selected:r27.selected,baseline:r26.baseline_appearance,selected26:r26.selected_appearance,baselineRaw,rightEye:research.right_eye_diagnostics};
await fs.writeFile(path.join(output,'.build/report_data.json'),JSON.stringify(payload,null,2));
console.log(JSON.stringify({output,workItems:work.length,links:links.length,httpFailures:links.filter(r=>r[6]!==200),executions:executions.length,succeeded:executions.filter(r=>r[6]==='성공').length,failed:executions.filter(r=>r[6]==='실패').length,appearances:appearances.length,surfaceCandidates:surface.length}));
