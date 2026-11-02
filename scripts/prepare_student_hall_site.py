#!/usr/bin/env python3
"""Prepare separate student-union results without replacing older RGB/NIR results."""
from pathlib import Path
import json,shutil,subprocess
import numpy as np
import open3d as o
import pycolmap
from PIL import Image
from reconstruct_student_hall_lidar import OUT,transform
from improve_stereo_reconstruction import write_ply

SITE=Path('sites/reconstruction-lab-share/dist');ASSETS=SITE/'assets';FUSED=OUT/'fused'

def geometry():
    stats=json.loads((FUSED/'summary.json').read_text());ps=np.load(FUSED/'selected_poses.npz')['poses']
    pc=o.io.read_point_cloud(str(FUSED/'lidar_full_map.ply'));full=np.asarray(pc.points);fc=np.uint8(np.asarray(pc.colors)*255+.5)
    q=np.load(FUSED/'student_hall_colored.npz');xyz,rgb=q['xyz'],q['rgb']
    base=[];bc=[]
    for c in [0,1]:
        r=pycolmap.Reconstruction(f'experiments/rgb_colmap_well_lit_outdoor_768_exhaustive_fixed/sparse/{c}')
        z=np.load(OUT/f'colmap_component{c}.npz');anchor=np.linalg.inv(z['poses'][0]);scale=z['scale']
        base.extend(transform(np.array([p.xyz for p in r.points3D.values()])*scale,anchor));bc.extend([p.color for p in r.points3D.values()])
    base,bc=np.asarray(base),np.asarray(bc)
    write_ply(FUSED/'colmap_original_metric.ply',base,bc)
    mesh=o.io.read_triangle_mesh(str(FUSED/'student_hall_poisson.ply'))
    roi_pose=ps[26];w2r=np.linalg.inv(roi_pose)
    def roi(x):
        c=transform(x,w2r)
        return (c[:,2]>-6)&(c[:,2]<100)&(np.abs(c[:,0])<32)&(c[:,1]>-32)&(c[:,1]<9)
    areas=[];rng=np.random.default_rng(42)
    for area,mask in [('entrance',roi),('route',lambda x:np.ones(len(x),bool))]:
        xx=xyz[mask(xyz)];cc=rgb[mask(xyz)];ff=full[mask(full)];fcc=fc[mask(full)]
        bb=base[mask(base)];bcc=bc[mask(base)]
        lo,hi=np.percentile(ff,[1,99],axis=0);center=(lo+hi)/2;radius=float(np.max((hi-lo)/2))
        offset=0 if area=='entrance' else 1
        for seg,x,c in [(10+offset,xx,cc),(12+offset,bb,bcc),(14+offset,ff,fcc)]:
            ids=rng.choice(len(x),min(350000,len(x)),replace=False)
            ((x[ids]-center)/radius).astype('<f4').tofile(ASSETS/f's{seg}-points.bin');c[ids].astype('u1').tofile(ASSETS/f's{seg}-colors.bin')
            write_ply(ASSETS/f's{seg}-sample.ply',x[ids],c[ids])
        m=o.geometry.TriangleMesh(mesh)
        if area=='entrance':m.remove_vertices_by_mask(~mask(np.asarray(m.vertices)))
        if len(m.triangles)>180000:m=m.simplify_quadric_decimation(180000)
        m.compute_vertex_normals();seg=10+offset
        ((np.asarray(m.vertices)-center)/radius).astype('<f4').tofile(ASSETS/f's{seg}-mesh-vertices.bin')
        np.asarray(m.vertex_normals).astype('<f4').tofile(ASSETS/f's{seg}-mesh-normals.bin')
        np.uint8(np.clip(np.asarray(m.vertex_colors),0,1)*255+.5).tofile(ASSETS/f's{seg}-mesh-colors.bin')
        np.asarray(m.triangles).astype('<u4').tofile(ASSETS/f's{seg}-mesh-faces.bin')
        o.io.write_triangle_mesh(str(ASSETS/f's{seg}-mesh.ply'),m)
        areas.append({'id':area,'segment':seg,'points':len(xx),'lidarPoints':len(ff),'baselinePoints':len(bb),
            'triangles':len(m.triangles),'radius_m':radius,'center':center.tolist()})
    photos=[]
    for i in [0,6,16,20,26,32,38,44,47]:
        Image.open(OUT/'images'/f'{i:06d}.png').save(ASSETS/f'hall-photo-{i}.jpg',quality=91)
        photos.append({'frame':i,'src':f'assets/hall-photo-{i}.jpg'})
    Image.open(OUT/'images/000026.png').save(ASSETS/'hall-hero.jpg',quality=94)
    ev=json.loads((OUT/'evaluation.json').read_text());joint=json.loads((OUT/'joint_report.json').read_text())
    surf=json.loads((FUSED/'poisson_validation.json').read_text());tsdf=json.loads((FUSED/'mesh_validation.json').read_text())
    result={'summary':stats,'areas':areas,'photos':photos,'evaluation':ev['metrics'],'joint':joint,
        'meshValidation':surf,'tsdfValidation':tsdf,'baselinePoints':len(base),'sourceFrames':48,
        'limitations':['친구의 완성 지도를 받은 결과가 아니라, 데이터셋의 원시 LiDAR 48개를 직접 정합한 결과입니다.',
            '회색 표면은 LiDAR로 관측했지만 사진 색상을 확인하지 못한 영역입니다. 구멍·잘림·매끄럽게 뭉개진 부분이 남아 있습니다.',
            '수치는 센서 간·스캔 간 일관성입니다. 별도 실측 정답으로 건물 전체 치수 정확도를 검증하지 않았습니다.',
            'Poisson 표면은 점 사이를 추정해 연결합니다. 일부 계단·난간·식생·유리 구조는 부정확할 수 있습니다.']}
    (ASSETS/'student-hall.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    (SITE/'student-hall-report.md').write_text(Path('STUDENT_HALL_LIDAR_REPORT.md').read_text() if Path('STUDENT_HALL_LIDAR_REPORT.md').exists() else 'Report pending')
    print('student_hall_assets',json.dumps({'areas':areas,'summary':stats}),flush=True)

def rendering():
    path=ASSETS/'student-hall.json';r=json.loads(path.read_text());rows=[]
    for method in ['sparse','lidar','colmap']:
        for p in OUT.glob(f'gs_{method}_*/metrics.json'):
            metric=json.loads(p.read_text());metric['folder']=str(p.parent);rows.append(metric)
    if not rows:return
    chosen=max(rows,key=lambda x:x['metrics']['psnr']);folder=Path(chosen['folder'])
    # Avoid publishing internal filesystem paths.
    r['rendering']={k:v for k,v in chosen.items() if k!='folder'}
    r['renderingTrials']=[{k:v for k,v in x.items() if k!='folder'} for x in rows]
    Image.open(folder/'novel_032.jpg').save(ASSETS/'hall-render-poster.jpg',quality=93)
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-framerate','12','-i',str(folder/'novel_%03d.jpg'),
        '-c:v','libx264','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(ASSETS/'hall-flythrough.mp4')],check=True)
    for k,i in enumerate([20,26,32,38,44]):
        Image.open(folder/f'heldout_{k:02d}.png').save(ASSETS/f'hall-eval-{k}-render.jpg',quality=93)
        Image.open(OUT/'gs_images'/f'{i:06d}.png').save(ASSETS/f'hall-eval-{k}-input.jpg',quality=93)
    path.write_text(json.dumps(r,ensure_ascii=False,indent=2));print('rendering_selected',r['rendering'],flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['geometry','rendering']);a=p.parse_args()
    globals()[a.stage]()
