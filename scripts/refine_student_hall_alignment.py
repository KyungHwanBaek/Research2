#!/usr/bin/env python3
"""Metric alignment of expanded stereo SfM, with held-out anchor diagnostics.

Align only within the existing sensor coordinate system. This is not an
independent accuracy evaluation or an end-to-end tightly coupled SLAM system.
"""
from pathlib import Path
import json
import numpy as np
import pycolmap
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from reconstruct_student_hall_lidar import umeyama,transform
from improve_stereo_reconstruction import write_ply

B=Path('experiments/student_hall_lidar');O=Path('experiments/student_hall_refinement_20261026')
def main():
    rec=pycolmap.Reconstruction(O/'stereo_colmap/sparse/0');P=np.load(B/'fused/selected_poses.npz')['poses'];K=np.load(B/'calibration.npz')['K']
    refs=[];old=[];rots=[];dep={}
    for im in rec.images.values():
        if not im.name.startswith('left/'):continue
        i=int(Path(im.name).stem);T=np.eye(4);T[:3]=im.cam_from_world().matrix();pose=np.linalg.inv(T);refs.append(pose[:3,3]);old.append(P[i,:3,3]);rots.append(P[i,:3,:3]@pose[:3,:3].T)
        dep[i]=np.load(B/'frames'/f'{i:06d}.npz')['depth']
    R0=Rotation.from_matrix(rots).mean().as_matrix();ref=np.array(refs)@R0.T;old=np.array(old)
    scale=np.sum((ref-ref.mean(0))*(old-old.mean(0)))/np.sum((ref-ref.mean(0))**2);t0=old.mean(0)-scale*ref.mean(0)
    points=[];rgb=[];ids=[];timestamps=[];ax=[];ay=[];pid=[]
    for p_id,p in rec.points3D.items():
        times=set();targets=[]
        for e in p.track.elements:
            im=rec.images[e.image_id];i=int(Path(im.name).stem);times.add(i)
            if not im.name.startswith('left/'):continue
            uv=im.points2D[e.point2D_idx].xy*(2/3);ix,iy=np.rint(uv).astype(int)
            if not 1<=ix<959 or not 1<=iy<719:continue
            d=dep[i][iy,ix]
            if not 3<d<30:continue
            patch=dep[i][iy-1:iy+2,ix-1:ix+2];nz=patch[patch>0]
            if len(nz)<7 or np.ptp(nz)>(.08+.02*d):continue
            x=np.r_[(uv-K[:2,2])/[K[0,0],K[1,1]]*d,d];targets.append(transform(x[None],P[i])[0])
        points.append(p.xyz);rgb.append(p.color);ids.append(p_id);timestamps.append(len(times))
        if targets:
            target=np.median(targets,axis=0)
            if np.linalg.norm(p.xyz@R0.T*scale+t0-target)<1.5:ax.append(p.xyz);ay.append(target);pid.append(p_id)
    ax=np.array(ax);ay=np.array(ay);pid=np.array(pid);check=np.arange(len(ax))%5==0;train=~check
    def unpack(v):return np.exp(v[0]),Rotation.from_rotvec(v[1:4]).as_matrix()@R0,v[4:7]
    def residual(v):
        s,r,t=unpack(v);return ((ax[train]@r.T*s+t)-ay[train]).ravel()
    fit=least_squares(residual,np.r_[np.log(scale),np.zeros(3),t0],loss='soft_l1',f_scale=.12,max_nfev=150)
    s,r,t=unpack(fit.x);x=np.array(points)@r.T*s+t;rgb=np.array(rgb,np.uint8)
    errors=np.linalg.norm(ax@r.T*s+t-ay,axis=1)
    full=np.load(B/'fused/student_hall_colored.npz');lidar_xyz=full['xyz'];dist,_=cKDTree(lidar_xyz).query(x,workers=4)
    keep=(dist<.20)&(np.array(timestamps)>=3)
    xyz=np.vstack((lidar_xyz,x[keep]));color=np.vstack((full['rgb'],rgb[keep]));_,unique=np.unique(np.floor(xyz/.02).astype(int),axis=0,return_index=True)
    xyz,color=xyz[unique],color[unique]
    write_ply(O/'expanded_colmap_metric.ply',x,rgb);write_ply(O/'supported_fusion.ply',xyz,color)
    np.savez_compressed(O/'expanded_colmap_metric.npz',xyz=x.astype(np.float32),rgb=rgb,scale=s,R=r,t=t,supported=keep)
    np.savez_compressed(O/'supported_fusion.npz',xyz=xyz.astype(np.float32),rgb=color)
    report=dict(sfm_points=len(x),alignment_fit_landmarks=int(train.sum()),alignment_check_landmarks=int(check.sum()),
        alignment_check_median_cm=float(np.median(errors[check])*100),alignment_check_p90_cm=float(np.percentile(errors[check],90)*100),
        metric_scale=float(s),supported_new_points=int(keep.sum()),fused_colored_points=len(xyz),previous_colored_points=len(lidar_xyz),
        selection='At least 3 distinct timestamps and within 20cm of the existing photo-colored LiDAR cloud; 2cm voxel deduplication.',
        caveat='Approximate similarity alignment using LiDAR-backed feature depths. Anchor-check landmarks excluded from alignment fit, but source poses/SfM already used all captures. Not independent absolute accuracy. New SfM points are triangulated photo observations, not LiDAR returns.')
    (O/'alignment_report.json').write_text(json.dumps(report,indent=2));print('STEREO_LIDAR_ALIGNMENT',json.dumps(report),flush=True)
if __name__=='__main__':main()
