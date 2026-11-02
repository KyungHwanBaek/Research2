#!/usr/bin/env python3
"""Create reviewable geometry assets, evidence, metrics and downloadable files."""
from pathlib import Path
import json,hashlib,shutil,zipfile,subprocess
import numpy as np,open3d as o
from PIL import Image,ImageDraw
from scipy.spatial import cKDTree
from render_student_hall_geometry import render
ROOT=Path(__file__).resolve().parents[1];O=ROOT/'experiments/student_hall_geometry_20261031';SITE=ROOT/'sites/reconstruction-lab-share/dist';A=SITE/'assets';B=ROOT/'experiments/student_hall_lidar'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for part in iter(lambda:f.read(8*1024*1024),b''):h.update(part)
    return h.hexdigest()
def main():
    target=ROOT/'outputs/student_hall_geometry_20261031';target.mkdir(parents=True,exist_ok=True);downloads=SITE/'downloads';downloads.mkdir(exist_ok=True)
    P=np.load(B/'fused/selected_poses.npz');ps,K=P['poses'],P['K'];R=ps[26,:3,:3];t=ps[26,:3,3]
    def roi(x):
        q=(x-t)@R;return (q[:,2]>-4)&(q[:,2]<75)&(np.abs(q[:,0])<26)&(q[:,1]>-26)&(q[:,1]<8)
    definitions=[
      (100,'사진에서 만든 표면',O/'mvs_stereo_refined_final/photo_surface.ply','photo_surface.ply','photo','52장 RGB 사진 → 다중 시점 깊이 → 점군 → 메시 → 사진 대응에 따른 정점 보정. LiDAR는 좌표 정렬에 사용했습니다. 거친 면·구멍·잘못 연결된 면이 남은 실험 결과입니다.'),
      (101,'기존 구조 기준',ROOT/'experiments/student_hall_refinement_20261026/surfaces/baseline_recolored_final.ply','baseline_surface.ply','baseline','기존 LiDAR 표면에 사진 색상을 입힌 기준 모델입니다. 이번 실험에서 이를 확실하게 능가하는 구조는 확인하지 못했습니다.'),
      (102,'사진 점을 더한 후보',O/'hybrid_fixed_high_support0.05_final/surface_colored.ply','hybrid_surface.ply','hybrid','학습용 LiDAR에서 5cm 이내이고 방향도 일치하는 사진 복원 점만 추가했습니다. 두 검증 지표가 함께 개선되지 않아 교체하지 않은 후보입니다.'),
      (103,'다중 시점 학습 모델 후보',O/'neural_da3largecal_final_504_corrected_final_v0.08_s0_r0.025/surface.ply','learned_surface.ply','learned','DA3 Large의 사진 깊이를 LiDAR로 보정한 표면입니다. 실제 측정된 완성 형상이 아니며, 누락과 깊이 편향이 남아 있습니다.')]
    records=[];meshes=[]
    for seg,label,path,filename,kind,note in definitions:
        assert path.exists(),str(path);mesh=o.io.read_triangle_mesh(str(path));assert len(mesh.triangles)>0
        if kind=='photo':
            pc=o.io.read_point_cloud(str(O/'mvs_stereo_final/photo_dense_metric.ply'));_,j=cKDTree(np.asarray(pc.points)).query(np.asarray(mesh.vertices),workers=4)
            mesh.vertex_colors=o.utility.Vector3dVector(np.asarray(pc.colors)[j])
        if not mesh.has_vertex_colors():mesh.paint_uniform_color([.64,.67,.69])
        mesh.compute_vertex_normals();mesh.compute_triangle_normals();o.io.write_triangle_mesh(str(target/filename),mesh)
        meshes.append((label,mesh));preview=o.geometry.TriangleMesh(mesh);preview.remove_vertices_by_mask(~roi(np.asarray(preview.vertices)))
        preview=preview.simplify_quadric_decimation(150000);preview.compute_vertex_normals();x=np.asarray(preview.vertices)
        records.append(dict(segment=seg,label=label,kind=kind,note=note,vertices=len(mesh.vertices),triangles=len(mesh.triangles),preview_vertices=len(x),preview_triangles=len(preview.triangles),full_file=filename,preview_file=f'assets/s{seg}-mesh.ply',coordinate_system='Existing metric camera-zero world frame, metres'))
        # The same rigid coordinate transform and normalization are used for all
        # displayed candidates. Full PLY download retains original metre units.
        display=(x-t)@R;center=np.array([0.,-5.,32.]);radius=42.
        ((display-center)/radius).astype('<f4').tofile(A/f's{seg}-mesh-vertices.bin');(np.asarray(preview.vertex_normals)@R).astype('<f4').tofile(A/f's{seg}-mesh-normals.bin');np.uint8(np.clip(np.asarray(preview.vertex_colors),0,1)*255+.5).tofile(A/f's{seg}-mesh-colors.bin');np.asarray(preview.triangles).astype('<u4').tofile(A/f's{seg}-mesh-faces.bin');o.io.write_triangle_mesh(str(A/f's{seg}-mesh.ply'),preview,write_vertex_normals=False)
        # Point mode uses the actual dense photogrammetric points when available.
        if kind=='photo':x=np.asarray(pc.points);colors=np.asarray(pc.colors);keep=roi(x);x,colors=x[keep],colors[keep]
        else:x=np.asarray(preview.vertices);colors=np.asarray(preview.vertex_colors)
        if len(x)>250000:ids=np.random.default_rng(42).choice(len(x),250000,replace=False);x,colors=x[ids],colors[ids]
        (((x-t)@R-center)/radius).astype('<f4').tofile(A/f's{seg}-points.bin');np.uint8(np.clip(colors,0,1)*255+.5).tofile(A/f's{seg}-colors.bin')
    # Colour-free evidence makes geometry changes visible without texture.
    evidence=Image.new('RGB',(1440,390*3),'white')
    for row,(label,mesh) in enumerate([meshes[1],meshes[0],meshes[2]]):
        gray=o.geometry.TriangleMesh(mesh);gray.paint_uniform_color([.64,.67,.70]);gray.compute_triangle_normals()
        for col,i in enumerate([26,34,40]):
            im=Image.fromarray(render(gray,ps[i],K)).resize((480,360));evidence.paste(im,(col*480,row*390+30));ImageDraw.Draw(evidence).text((col*480+10,row*390+9),f'{["Baseline","Photographs","Photo+LiDAR candidate"][row]} | camera {i}',fill='black')
    evidence.save(O/'final_geometry_comparison.jpg',quality=95);evidence.save(A/'hall31-geometry-comparison.jpg',quality=92)
    Image.open(B/'images/000026.png').save(A/'hall31-original.jpg',quality=94)
    photo=o.geometry.TriangleMesh(meshes[0][1]);photo.paint_uniform_color([.64,.67,.7]);photo.compute_triangle_normals();Image.fromarray(render(photo,ps[26],K)).save(A/'hall31-photo-surface.jpg',quality=94)
    table=[]
    for p in sorted(O.glob('*/surface_report.json')):
        d=json.loads(p.read_text());v=d.get('validation');
        if not v:continue
        table.append(dict(name=p.parent.name,triangles=d['triangles'],hall_interpolated_mae_cm=v['hall_interpolated_mae_cm'],hall_raw_ray_mae_cm=v['raw_rays']['hall_raw_ray_mae_cm']))
    base=json.loads((B/'fused/poisson_validation.json').read_text());best=next(d for d in base['candidates'] if d['depth']==base['selected_depth']);raw=json.loads((ROOT/'experiments/student_hall_refinement_20261027/baseline_raw_ray_validation.json').read_text())
    baseline=dict(hall_interpolated_mae_cm=best['student_hall_mae_cm'],hall_raw_ray_mae_cm=raw['hall_raw_ray_mae_cm'])
    finalphoto=json.loads((O/'mvs_stereo_refined_final/surface_report.json').read_text());overlap=json.loads((O/'additional_outdoor_overlap.json').read_text())
    report=dict(date_kst='2026-10-31',goal='Photograph-based 3D surface reconstruction',outcome='Photo-derived dense point clouds and actual meshes produced; clear structural-accuracy gain over the existing measured baseline not demonstrated.',models=records,baseline=baseline,validation_candidates=table,photo_final=finalphoto,additional_sequences=[dict(sequence='01-26-15-47-20',frames=112,used_in_mesh=False,reason='Interior route; not registered to the existing exterior reconstruction.'),dict(sequence='01-26-15-53-03',frames=277,used_in_mesh=False,screened_pairs=overlap['pairs_tested'],max_geometric_inliers=max([r['geometric_inliers'] for r in overlap['top_pairs']],default=0),reason='No reliable shared reconstruction established by initial visual/geometric screening; no unverified scene merge.')],sources=['https://www.shbaek.com/','https://divisonofficer.github.io/project_page_Pixel_aligned_RGB_NIR_Stereo/','https://github.com/divisonofficer/Pixel_aligned_RGB_NIR_Stereo','https://huggingface.co/datasets/DivisonOfficer/Pixel-aligned_RGB-NIR_stereo_dataset','https://github.com/princeton-vl/RAFT-Stereo','https://github.com/cdcseacave/openMVS','https://github.com/ByteDance-Seed/Depth-Anything-3'],evaluation_note='Six whole LiDAR scans and their photographs excluded from validation-surface construction. Four hall views scored. SfM and global sensor poses use all captures, and repeated candidate selection reuses validation. Errors capped at 1m and missing surfaces penalized at 1m. Interpolated camera depths and raw physical LiDAR rays are separate metrics; neither is surveyed absolute building accuracy. Photo-only methods have narrower observed coverage than full 360-degree LiDAR. Public POSTECH checkpoint training overlap with this dataset is not excluded.')
    (target/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));(A/'hall31-results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));(O/'results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    # Preserve evidence that old models/poses and raw inputs still match prior
    # recorded manifests. Do not include raw photos or pretrained weights in ZIP.
    manifest=json.loads((ROOT/'experiments/student_hall_refinement_20261027/baseline_manifest.json').read_text());(O/'prior_baseline_manifest_reference.json').write_text(json.dumps(manifest,indent=2))
    for name in ['run_hall_geometry.py','hall_geometry_mvs.py','hall_geometry_depth.py','hall_geometry_surface.py','hall_geometry_rig.py','hall_geometry_da3.py','hall_geometry_calibrate.py','hall_geometry_hybrid.py','hall_geometry_publish.py','hall_geometry_finish.py']:
        dst=O/'source_snapshot'/name;dst.parent.mkdir(exist_ok=True);shutil.copy2(ROOT/'scripts'/name,dst)
    env={}
    for name in ['.venv','.venv-gs-refine','.venv-mesh312']:
        env[name]=subprocess.check_output([str(ROOT/name/'bin/python'),'-m','pip','freeze'],text=True)
    (O/'environment.json').write_text(json.dumps(env,indent=2));provenance={}
    for name in ['pixel_aligned_rgb_nir','raft_stereo','openmvs_source','depth_anything_3']:
        provenance[name]=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT/'third_party'/name,text=True).strip()
    for name in ['third_party/pixel_aligned_rgb_nir/weights/model_feature_fusion.pth','third_party/raft_stereo/raftstereo-middlebury.pth','third_party/depth_anything_3/weights/DA3-SMALL/model.safetensors','third_party/depth_anything_3/weights/DA3-LARGE/model.safetensors']:
        provenance[name]=sha(ROOT/name)
    (O/'provenance.json').write_text(json.dumps(provenance,indent=2))
    # Public copy strips private local filesystem paths from experiment details.
    safe=json.dumps(report,ensure_ascii=False,indent=2).replace(str(ROOT),'/PROJECT');(A/'hall31-results.json').write_text(safe)
    publiczip=downloads/'student_hall_geometry_20261031_preview.zip'
    with zipfile.ZipFile(publiczip,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for r in records:z.write(SITE/r['preview_file'],r['full_file'].replace('.ply','_preview.ply'))
        z.writestr('results.json',safe);z.write(O/'final_geometry_comparison.jpg','geometry_comparison.jpg')
    print(json.dumps(dict(output=str(target),models=records,baseline=baseline,public_zip_bytes=publiczip.stat().st_size),ensure_ascii=False),flush=True)
if __name__=='__main__':main()
