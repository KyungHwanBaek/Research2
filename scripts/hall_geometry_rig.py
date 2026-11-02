#!/usr/bin/env python3
"""Photogrammetric BA with a physical, fixed stereo rig and metric baseline."""
from pathlib import Path
import argparse,json,sqlite3
import numpy as np,pycolmap
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/student_hall_geometry_20261031'
SRC=ROOT/'experiments/student_hall_refinement_20261026/stereo_colmap'
def main(a):
    folder=OUT/('sfm_rig_refined' if a.refine else 'sfm_rig_fixed');folder.mkdir(exist_ok=True)
    dbpath=folder/'database.db'
    if not dbpath.exists():
        src=sqlite3.connect('file:'+str(SRC/'database.db')+'?mode=ro',uri=True);dst=sqlite3.connect(dbpath);src.backup(dst);dst.close();src.close()
    rec=pycolmap.Reconstruction(SRC/'sparse/0');images={im.name:im for im in rec.images.values()};baseline=[]
    for im in images.values():
        if im.name.startswith('left/'):baseline.append(np.linalg.norm(images['right/'+Path(im.name).name].projection_center()-im.projection_center()))
    c=np.load(ROOT/'experiments/student_hall_refinement_20261026/stereo_calibration.npz');R,T=c['R'],c['T'].ravel();scale=np.linalg.norm(T)/np.median(baseline)
    rec.transform(pycolmap.Sim3d(scale,pycolmap.Rotation3d(),np.zeros(3)))
    left=pycolmap.RigConfigCamera(ref_sensor=True,image_prefix='left/');right=pycolmap.RigConfigCamera(ref_sensor=False,image_prefix='right/',cam_from_rig=pycolmap.Rigid3d(pycolmap.Rotation3d(R),T))
    rig=pycolmap.RigConfig(cameras=[left,right])
    with pycolmap.Database.open(dbpath) as database:pycolmap.apply_rig_config([rig],database,rec)
    config=pycolmap.BundleAdjustmentConfig()
    for im in rec.images.values():config.add_image(im.image_id)
    first=min(rec.images.values(),key=lambda im:im.name);config.set_constant_rig_from_world_pose(first.frame_id)
    for rig in rec.rigs.values():
        for sensor in rig.non_ref_sensors:config.set_constant_sensor_from_rig_pose(sensor)
    opts=pycolmap.BundleAdjustmentOptions();opts.refine_focal_length=a.refine;opts.refine_principal_point=a.refine;opts.refine_extra_params=False;opts.refine_sensor_from_rig=False
    opts.ceres.loss_function_type=pycolmap.LossFunctionType.SOFT_L1;opts.ceres.loss_function_scale=1.;opts.ceres.solver_options.max_num_iterations=150;opts.ceres.solver_options.num_threads=4
    before=rec.compute_mean_reprojection_error();adjuster=pycolmap.create_default_bundle_adjuster(opts,config,rec);summary=adjuster.solve();rec.update_point_3d_errors();rec.write(folder)
    rows=[]
    for im in rec.images.values():
        if not im.name.startswith('left/'):continue
        other=next(v for v in rec.images.values() if v.name=='right/'+Path(im.name).name);rows.append(np.linalg.norm(other.projection_center()-im.projection_center()))
    report=dict(images=rec.num_reg_images(),points=rec.num_points3D(),rigs=rec.num_rigs(),frames=rec.num_frames(),baseline_m=float(np.linalg.norm(T)),baseline_min_max_m=[float(np.min(rows)),float(np.max(rows))],intrinsics_refined=a.refine,mean_reprojection_error_before_px=float(before),mean_reprojection_error_after_px=float(rec.compute_mean_reprojection_error()),initial_metric_scale=float(scale),note='Fixed physical RGB stereo rig. Camera poses and 3D landmarks optimized jointly; LiDAR depths and map not in BA. Raw dataset calibration plus earlier small stereo extrinsic refinement is reused. Source images include validation captures for SfM poses only.')
    (folder/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--refine',action='store_true');main(p.parse_args())
