#!/usr/bin/env python3
"""Photo-only CPU MVS. LiDAR is used only for a recorded world-frame alignment.

Validation excludes all photos of six withheld scans from MVS fusion. SfM
poses still use all photographs, so this is not a fully independent test.
"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import pycolmap
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from run_hall_geometry import ROOT,OUT,execute
HOLD=[5,13,21,29,37,45]
PREV=ROOT/'experiments/student_hall_refinement_20261027'
BASE=ROOT/'experiments/student_hall_lidar'
def prepare(mode, variant):
    source=(OUT/'sfm_rig_refined' if variant.startswith('rig') else PREV/'sfm_ba_principal' if variant.startswith(('principal','stereo')) else ROOT/'experiments/student_hall_refinement_20261026/stereo_colmap/sparse/0')
    folder=OUT/f'mvs_{variant}_{mode}';sparse=folder/'colmap/sparse';sparse.mkdir(parents=True,exist_ok=True)
    rec=pycolmap.Reconstruction(source)
    original_images=list(rec.images.values());P=np.load(BASE/'fused/selected_poses.npz')['poses']
    # Robust similarity alignment from camera poses, with no target mesh used.
    src=[];dst=[];rots=[];baselines=[]
    eyes={im.name:im for im in original_images}
    for im in original_images:
        if not im.name.startswith('left/'): continue
        i=int(Path(im.name).stem);T=np.eye(4);T[:3]=im.cam_from_world().matrix();Q=np.linalg.inv(T)
        src.append(Q[:3,3]);dst.append(P[i,:3,3]);rots.append(P[i,:3,:3]@Q[:3,:3].T)
        other=eyes.get('right/'+Path(im.name).name)
        if other is not None:baselines.append(np.linalg.norm(other.projection_center()-im.projection_center()))
    src=np.array(src);dst=np.array(dst);R0=Rotation.from_matrix(rots).mean().as_matrix();a=src@R0.T
    s0=np.sum((a-a.mean(0))*(dst-dst.mean(0)))/np.sum((a-a.mean(0))**2);t0=dst.mean(0)-s0*a.mean(0)
    if variant.startswith('rig'):s0=1.;t0=dst.mean(0)-a.mean(0)
    def unpack(v):return (1. if variant.startswith('rig') else np.exp(v[0])),Rotation.from_rotvec(v[1:4]).as_matrix()@R0,v[4:7]
    def residual(v):
        s,r,t=unpack(v)
        # Almost collinear camera centres cannot constrain roll around the
        # trajectory. Preserve the independently observed camera orientations.
        orientation=Rotation.from_matrix(np.asarray(rots)@r.T).as_rotvec()
        return np.r_[((src@r.T*s+t-dst)/.10).ravel(),(orientation/.01).ravel(),v[0] if variant.startswith('rig') else 0.]
    fit=least_squares(residual,np.r_[np.log(s0),np.zeros(3),t0],loss='soft_l1',f_scale=1.)
    s,r,t=unpack(fit.x);err=np.linalg.norm(src@r.T*s+t-dst,axis=1)
    np.savez(folder/'alignment.npz',scale=s,R=r,t=t)
    if mode=='validation':
        for im in original_images:
            if int(Path(im.name).stem) in HOLD and rec.frame(im.frame_id).has_pose():rec.deregister_frame(im.frame_id)
    rec.write(sparse)
    link=folder/'colmap/images'
    if not link.exists():link.symlink_to(ROOT/'experiments/student_hall_refinement_20261026/stereo_colmap/images',target_is_directory=True)
    row=dict(source=str(source.relative_to(ROOT)),mode=mode,variant=variant,images=rec.num_reg_images(),points=rec.num_points3D(),heldout_scans=HOLD if mode=='validation' else [],scale=float(s),camera_alignment_median_cm=float(np.median(err)*100),camera_alignment_p90_cm=float(np.percentile(err,90)*100),stereo_baseline_median_m=float(np.median(baselines)*s),stereo_baseline_p10_p90_m=(np.percentile(baselines,[10,90])*s).tolist(),note='Surface geometry is estimated from photographs only. Similarity uses existing LiDAR-derived camera poses; all-photo SfM poses reused. Camera residual is not a measured surface accuracy.')
    (folder/'preparation.json').write_text(json.dumps(row,indent=2));print(json.dumps(row),flush=True)
    return folder
def main(a):
    folder=OUT/f'mvs_{a.variant}_{a.mode}'
    if a.stage=='prepare':prepare(a.mode,a.variant);return
    binary=ROOT/'third_party/openmvs'
    if a.stage=='import':
        execute(f'mvs_{a.variant}_{a.mode}_import',[binary/'InterfaceCOLMAP','-i',folder/'colmap','-o',folder/'scene.mvs','--image-folder',folder/'colmap/images','--max-threads','4'])
    elif a.stage=='dense':
        high=a.variant.endswith('_high')
        extra=[]
        if a.variant=='stereo':
            config=folder/'narrow_baseline.cfg';config.write_text('Min Angle = 0.3\nOptim Angle = 3.0\nMin Area = 0.03\n');extra=['--dense-config-file',config]
        execute(f'mvs_{a.variant}_{a.mode}_dense',[binary/'DensifyPointCloud','-i',folder/'scene.mvs','-o',folder/'scene_dense.mvs','-w',folder,'--resolution-level','0' if high else '1','--max-resolution','1440' if high else '960','--min-resolution','640','--number-views','8' if high else '6','--number-views-fuse','2' if high or a.variant=='stereo' else '3','--iters','4','--geometric-iters','2','--max-threads','6','--estimate-roi','0','--crop-to-roi','0','--tower-mode','0','--archive-type','2',*extra])
    elif a.stage=='mesh':
        execute(f'mvs_{a.variant}_{a.mode}_mesh',[binary/'ReconstructMesh','-i',folder/'scene_dense.mvs','-o',folder/'scene_mesh.mvs','-w',folder,'--max-threads','6','--close-holes','0','--smooth','0','--free-space-support','1','--crop-to-roi','0','--archive-type','2'])
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','import','dense','mesh']);p.add_argument('--mode',choices=['validation','final'],default='validation');p.add_argument('--variant',choices=['principal','fixed','principal_high','fixed_high','rig','stereo'],default='principal');main(p.parse_args())
