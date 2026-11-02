#!/usr/bin/env python3
"""Reuse measured results for an educational four-stage comparison page."""
from pathlib import Path
import json,numpy as np,open3d as o
ROOT=Path(__file__).resolve().parents[1];S=ROOT/'sites/reconstruction-lab-share/dist';A=S/'assets';B=ROOT/'experiments/student_hall_lidar/fused'
def main():
    P=np.load(B/'selected_poses.npz')['poses'][26];R,t=P[:3,:3],P[:3,3];rows=[]
    for seg,name,file in [(110,'colmap','colmap_original_metric.ply'),(111,'lidar','lidar_full_map.ply')]:
        p=o.io.read_point_cloud(str(B/file));x=np.asarray(p.points);colors=np.asarray(p.colors) if p.has_colors() else np.full_like(x,.69);q=(x-t)@R;keep=(q[:,2]>-4)&(q[:,2]<75)&(np.abs(q[:,0])<26)&(q[:,1]>-26)&(q[:,1]<8);ids=np.flatnonzero(keep)
        if len(ids)>120000:ids=np.random.default_rng(42).choice(ids,120000,replace=False)
        ((q[ids]-[0,-5,32])/42).astype('<f4').tofile(A/f's{seg}-points.bin');np.uint8(np.clip(colors[ids],0,1)*255+.5).tofile(A/f's{seg}-colors.bin');rows.append(dict(id=name,total_points=len(x),roi_points=int(keep.sum()),display_points=len(ids),segment=seg))
    data=dict(models=rows,scope='Student hall sequence 01-26-15-41-24 only',display_coordinates='All four results use the same camera-26 ROI and hall31 normalization. Original sparse COLMAP components were scaled/repositioned using LiDAR calibration for display; no LiDAR vertices were added to the sparse cloud.',note='The accumulated LiDAR cloud is not a separate LiDAR-only tracking experiment: its scan alignment used photo/COLMAP initialization and LiDAR registration. These are four result categories, not four interchangeable algorithms.')
    (A/'hall-overview.json').write_text(json.dumps(data,ensure_ascii=False,indent=2))
    # Reuse the camera-like perspective already tested on the latest result page.
    src=(S/'hall31.js').read_text();start=src.index('  class GeometryViewer');end=src.index('  const viewer=',start);cls=src[start:end]
    # Apply the gray override exclusively to colour buffers. A missing pair of
    # parentheses previously converted LiDAR position buffers to Uint8 gray data.
    old="return !$('color').checked&&url.endsWith('colors.bin')?"
    new="return url.endsWith('colors.bin')&&(selected==='lidar'||!$('color').checked)?"
    assert old in cls
    cls=cls.replace(old,new).replace('gl_PointSize=pointSize;','gl_PointSize=pointSize*3.;')
    cls=cls.replace('reset(){this.yaw=0;this.pitch=0;this.zoom=3.25;this.distance=32/42;', 'reset(){this.yaw=-.35;this.pitch=-.4;this.zoom=2.8;this.distance=2.4;')
    controller=(ROOT/'scripts/hall_overview_controller.js').read_text();(S/'hall-overview.js').write_text("'use strict';\n(async()=>{\nconst $=id=>document.getElementById(id);let selected='surface';\n"+cls+controller+'\n})();\n')
    print(json.dumps(data,ensure_ascii=False));print('siteMiB',sum(p.stat().st_size for p in S.rglob('*') if p.is_file())/2**20)
if __name__=='__main__':main()
