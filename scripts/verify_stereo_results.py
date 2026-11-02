#!/usr/bin/env python3
"""Verify projection geometry, held-out image indexing and exported artifacts."""
from pathlib import Path
import json
import re
import subprocess
import base64
import cv2
import numpy as np
import open3d as o3d
from PIL import Image,ImageDraw
from improve_stereo_reconstruction import Stereo


def main():
    output=Path('results/stereo_improved_0905')
    s=Stereo('data/extracted/sampled_540p/09-05-17-34-36','experiments/stereo_improved_0905/depth')
    d=s.combined_depth(300);yy,xx=np.where(d>0)
    uv=np.column_stack((xx[::500],yy[::500])).astype(np.float32)
    xyz=s.points(d,uv);projected=xyz[:,:2]/xyz[:,2:]*np.array([s.rectK[0,0],s.rectK[1,1]])+s.rectK[:2,2]
    roundtrip=float(np.max(np.abs(projected-uv)));assert roundtrip<1e-8
    z=np.load(output/'camera_poses.npz');rot=z['poses'][:,:3,:3]
    orth=float(np.max(np.abs(np.einsum('nji,njk->nik',rot,rot)-np.eye(3))))
    assert orth<1e-8 and np.allclose(np.linalg.det(rot),1)
    graph=json.load(open('experiments/stereo_improved_0905/optimized_poses.json'))
    assert graph['solver_success'] and graph['final_robust_cost']<graph['initial_robust_cost']
    stats=json.load(open(output/'geometry_metrics.json'));checks=[]
    for r in stats:
        seg=r['segment'];p=o3d.io.read_point_cloud(str(output/f'segment{seg}_points.ply'))
        m=o3d.io.read_triangle_mesh(str(output/f'segment{seg}_mesh.ply'))
        assert len(p.points)==r['final_points'] and np.isfinite(np.asarray(p.points)).all()
        assert len(m.triangles)==r['mesh_triangles'] and len(m.triangles)>0
        assert np.asarray(m.triangles).max()<len(m.vertices)
        checks.append({'segment':seg,'points':len(p.points),'vertices':len(m.vertices),'triangles':len(m.triangles)})
        for suffix in ['points','mesh']:
            html=(output/f'segment{seg}_{suffix}.html').read_text()
            script=re.search(r'<script>(.*?)</script>',html,re.S)[1]
            target=output/'viewer_check.js';target.write_text(script)
            subprocess.run(['node','--check',str(target)],check=True)
            data=json.loads(re.search(r'const data=(.*?);\nfunction unpack',script,re.S)[1])
            n=len(base64.b64decode(data['positions']))//12
            assert len(base64.b64decode(data['colors']))==n*3
            if 'faces' in data:
                triangles=np.frombuffer(base64.b64decode(data['faces']),dtype='<u4')
                assert triangles.max()<n
    selected=json.load(open('exports/3dgs_0905_comparison_sparse/manifest.json'))['images']
    manual=[]
    for j,i in enumerate(selected[::8]):
        a=np.asarray(Image.open(f'exports/3dgs_stereo_0905_images/{i:06d}.png'),float)/255
        b=np.asarray(Image.open(f'results/renders/3dgs_0905_sparse_30000/heldout_{j:03d}.png'),float)/255
        manual.append(float(-10*np.log10(np.mean((a-b)**2))))
    metric=json.load(open('results/renders/3dgs_0905_sparse_30000/metrics.json'))['heldout_metrics']
    assert abs(np.mean(manual)-metric['psnr'])<.03
    before=json.load(open('experiments/stereo_original_0905/depth/diagnostics.json'))
    after=json.load(open('experiments/stereo_improved_0905/depth/diagnostics.json'))
    calibration={'before_vertical_error_mean_px':float(np.mean([f['epipolar_vertical_median_px'] for f in before['frames']])),
        'after_vertical_error_mean_px':float(np.mean([f['epipolar_vertical_median_px'] for f in after['frames']])),
        'before_rgb_valid_fraction':float(np.mean([f['rgb_valid_fraction'] for f in before['frames']])),
        'after_rgb_valid_fraction':float(np.mean([f['rgb_valid_fraction'] for f in after['frames']]))}
    assert calibration['after_vertical_error_mean_px']<.4
    report={'projection_roundtrip_max_px':roundtrip,'camera_rotation_orthogonality_max_error':orth,
        'pose_graph_converged':graph['solver_success'],'tracking_camera_count':len(z['indices']),
        'tracking_segments':len(np.unique(z['components'])),'geometry':checks,'calibration':calibration,
        'heldout_psnr_from_saved_8bit_images':float(np.mean(manual)),'renderer_metrics':metric,
        'viewer_javascript_syntax_and_buffers':'passed','browser_visual_interaction':'unavailable: browser runtime returned no browsers',
        'ground_truth_geometry':'not available; absolute geometry accuracy is not established'}
    (output/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    # Inputs and outputs in a directly comparable held-out image grid.
    width,height=360,270;sheet=Image.new('RGB',(width*3,height*4+45),'#10151d')
    draw=ImageDraw.Draw(sheet)
    for col,text in enumerate(['Held-out input','Sparse init / 12,000','Selected / 30,000']):draw.text((col*width+12,16),text,fill='white')
    for row,j in enumerate([0,2,4,6]):
        i=selected[::8][j]
        paths=[f'exports/3dgs_stereo_0905_images/{i:06d}.png',f'results/renders/3dgs_0905_sparse_12000/heldout_{j:03d}.png',f'results/renders/3dgs_0905_sparse_30000/heldout_{j:03d}.png']
        for col,path in enumerate(paths):sheet.paste(Image.open(path).resize((width,height)),(col*width,row*height+45))
    sheet.save(output/'heldout_comparison.jpg',quality=94)


if __name__=='__main__':main()
