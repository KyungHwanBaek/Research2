#!/usr/bin/env python3
"""Prepare compact, real-data-only assets for the public reconstruction gallery."""
from pathlib import Path
import json,shutil,argparse,subprocess
import numpy as np
import open3d as o3d
from PIL import Image
from improve_stereo_reconstruction import write_ply


def main():
    p=argparse.ArgumentParser();p.add_argument('--geometry',default='experiments/stereo_improved_0905/fused')
    a=p.parse_args();source=Path(a.geometry);stats=json.loads((source/'summary.json').read_text())
    site=Path('sites/reconstruction-lab-share/dist');assets=site/'assets';assets.mkdir(exist_ok=True)
    frames=sorted(p for p in Path('data/extracted/sampled_540p/09-05-17-34-36').iterdir() if (p/'rgb/left.png').exists())
    result={'segments':[],'geometryMethod':'stereo_joint_bundle_adjustment' if 'joint' in str(source) else 'stereo_pose_graph',
        'limitations':['전체 경로는 세 구간으로 나뉘어 있으며, 구간 사이의 위치 관계는 정합되지 않았습니다.','거리 단위는 카메라 보정값에 따른 추정치이며, 실제 치수 정확도를 독립 측정으로 검증하지 않았습니다.']}
    titles=['도로와 복도를 따라','중정을 향한 회전','중정과 조형물 사이']
    descriptions=['도로, 기둥이 있는 복도와 나무 사이 길을 포함하는 가장 긴 이동 구간입니다.','건물 입구에서 넓은 중정 쪽을 바라보며 방향을 바꾸는 짧은 구간입니다.','중정 주변의 건물과 나무를 관찰하며 조형물이 있는 쪽까지 이동합니다.']
    for r in stats:
        seg=r['segment'];stem=source/f'stereo_rgb_nir_segment{seg}'
        z=np.load(stem.with_suffix('.npz'));xyz,rgb=z['xyz'],z['rgb']
        lo,hi=np.percentile(xyz,[1,99],axis=0);center=(lo+hi)/2;radius=float(np.max(np.percentile(np.abs(xyz-center),98,axis=0)))
        rng=np.random.default_rng(100+seg);ids=rng.choice(len(xyz),min(260000,len(xyz)),replace=False)
        ((xyz[ids]-center)/radius).astype('<f4').tofile(assets/f's{seg}-points.bin');rgb[ids].astype('u1').tofile(assets/f's{seg}-colors.bin')
        write_ply(assets/f's{seg}-sample.ply',xyz[ids],rgb[ids])
        mesh=o3d.io.read_triangle_mesh(str(stem)+'_tsdf_preview.ply')
        if len(mesh.triangles)>100000:mesh=mesh.simplify_quadric_decimation(100000)
        mesh.compute_vertex_normals()
        ((np.asarray(mesh.vertices)-center)/radius).astype('<f4').tofile(assets/f's{seg}-mesh-vertices.bin')
        np.asarray(mesh.vertex_normals).astype('<f4').tofile(assets/f's{seg}-mesh-normals.bin')
        np.uint8(np.clip(np.asarray(mesh.vertex_colors),0,1)*255).tofile(assets/f's{seg}-mesh-colors.bin')
        np.asarray(mesh.triangles).astype('<u4').tofile(assets/f's{seg}-mesh-faces.bin')
        photos=[]
        for j,i in enumerate(np.linspace(*r['source_frame_range'],6,dtype=int)):
            name=f's{seg}-photo{j}.jpg';im=Image.open(frames[int(i)]/'rgb/left.png').convert('RGB');im.save(assets/name,quality=88)
            photos.append({'src':'assets/'+name,'frame':int(i),'time':frames[int(i)].name.replace('_',':',2).replace('_','.')})
        result['segments'].append({'id':seg,'title':titles[seg],'description':descriptions[seg],
            'points':len(xyz),'triangles':r['mesh_triangles'],'displayPoints':len(ids),'displayTriangles':len(mesh.triangles),
            'frameRange':r['source_frame_range'],'photos':photos,'start':photos[0]['time'],'end':photos[-1]['time'],
            'download':f'assets/s{seg}-sample.ply'})
    selected=None;comparison=[]
    for method,iters in [('sparse',12000),('stereo',12000),('sparse',30000),('joint',30000)]:
        folder=Path('results/renders')/f'3dgs_0905_{method}_{iters}'
        if not (folder/'metrics.json').exists():continue
        data=json.loads((folder/'metrics.json').read_text());metric=data['heldout_metrics'];comparison.append({'method':method,'iterations':iters,**metric})
        if selected is None or metric['psnr']>selected[0]:selected=(metric['psnr'],folder,data)
    root=selected[1];result['rendering']={'method':selected[2]['method'],'iterations':selected[2]['iterations'],**selected[2]['heldout_metrics']}
    result['renderingComparison']=comparison
    Image.open(root/'novel_020.jpg').save(assets/'render-poster.jpg',quality=92)
    command=['ffmpeg','-hide_banner','-loglevel','error','-y','-framerate','12','-i',str(root/'novel_%03d.jpg'),'-c:v','libx264','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(assets/'flythrough.mp4')]
    subprocess.run(command,check=True)
    selected_frames=json.loads(Path('exports/3dgs_0905_comparison_sparse/manifest.json').read_text())['images'][::8]
    eval_photos=[]
    for j in [2,4,6]:
        i=selected_frames[j]
        for label,path in [('input',Path('exports/3dgs_stereo_0905_images')/f'{i:06d}.png'),('render',root/f'heldout_{j:03d}.png')]:
            Image.open(path).save(assets/f'eval-{j}-{label}.jpg',quality=92)
        eval_photos.append({'index':j,'frame':i})
    result['evaluationExamples']=eval_photos
    source_im=Image.open(frames[300]/'rgb/left.png').convert('RGB');source_im.save(assets/'hero.jpg',quality=94)
    result['totalPoints']=sum(r['points'] for r in result['segments']);result['totalTriangles']=sum(r['triangles'] for r in result['segments'])
    old=Path('experiments/stereo_improved_0905/fused/summary.json');oldstats=json.loads(old.read_text())
    result['previousTotalPoints']=sum(r['final_points'] for r in oldstats)
    comp=Path('experiments/stereo_joint_ba_0905/comparison.json')
    if comp.exists():result['jointBAComparison']=json.loads(comp.read_text())
    (assets/'scene.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    (site/'robots.txt').write_text('User-agent: *\nAllow: /\n')
    largest=max((p.stat().st_size,p.name) for p in assets.iterdir() if p.is_file())
    assert largest[0]<24*1024*1024,largest
    print('site_assets',len(list(assets.iterdir())),'largest',largest,'best_rendering',str(root),flush=True)


if __name__=='__main__':main()
