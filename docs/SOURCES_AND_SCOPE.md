# 외부 자료와 저장소 포함 범위

## 원 데이터

Jinnyeong Kim, Seung-Hwan Baek, *Pixel-aligned RGB-NIR Stereo Imaging and Dataset for Robot Vision*, CVPR 2025.

- https://divisonofficer.github.io/project_page_Pixel_aligned_RGB_NIR_Stereo/
- https://github.com/divisonofficer/Pixel_aligned_RGB_NIR_Stereo
- https://huggingface.co/datasets/DivisonOfficer/Pixel-aligned_RGB-NIR_stereo_dataset

공개 데이터에 기반한 프로젝트이며 사용자 별도 촬영이나 친구의 비공개 지도를 사용한 것으로 설명하지 않는다. 원 데이터는 이 Git 저장소에 재배포하지 않는다.

## 외부 도구

- COLMAP: https://github.com/colmap/colmap
- OpenMVS: https://github.com/cdcseacave/openMVS
- Open3D: https://github.com/isl-org/Open3D
- Mac Metal msplat: https://github.com/rayanht/msplat
- MetalSplat: https://github.com/tchauffi/metalsplat
- RAFT-Stereo: https://github.com/princeton-vl/RAFT-Stereo
- Depth Anything 3: https://github.com/ByteDance-Seed/Depth-Anything-3

외부 checkout의 실제 origin과 revision은 [third_party_versions.json](third_party_versions.json)에 보존한다. revision은 추적 가능한 출발점이며, 로컬 패치가 없었다는 의미는 아니다. Metal 런타임·처리 한도 조정은 프로젝트 보고서와 `hall_research_native.py`, `hall_research_runtime_build.py`에 기록되어 있다.

외부 소스·실행 파일·모델 가중치를 이 저장소로 복사하지 않는다. 다운로드·빌드·사용·재배포 시 각 프로젝트의 라이선스와 인용 조건을 따른다. 외부 도구를 사용했다는 사실은 해당 알고리즘의 저작권이나 독창성을 주장하는 것이 아니다.

## 포함·제외

포함: 프로젝트 작성 스크립트, 의존성 목록, Markdown 보고서, 작은 선택 평가 JSON, 재현 조건과 LLM 전달 요약.

제외: 가상환경, 원본 RGB/NIR/LiDAR, 대형 점군·메시·GLB·splat·학습 체크포인트, 외부 코드 checkout, 사이트의 별도 Git 저장소·배포 설정, 인증 정보, 로그 전체, 문서 바이너리.

일부 과거 보고서는 제외된 로컬 파일·실험 폴더를 참조한다. 최종 결과는 [공유 페이지](https://stereo-reconstruction-lab-kyunghwan.qj8scj68m5.chatgpt.site/student-hall-overview#viewer-section)에서 보거나 다운로드할 수 있다. 과거 보고서의 예전 공유 도메인은 보존 기록이며 현재 진입점은 위 링크다.
