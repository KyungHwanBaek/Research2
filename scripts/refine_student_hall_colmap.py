#!/usr/bin/env python3
"""Reconstruct student-hall RGB using both eyes at original resolution."""
from pathlib import Path
import argparse,json,subprocess,time
import cv2
import numpy as np
import pycolmap

ROOT=Path('experiments/student_hall_refinement_20261026')
OUT=ROOT/'stereo_colmap';SRC=Path('data/extracted/01-26-15-41-24')

def run(cmd):
    print('COMMAND',json.dumps(cmd),flush=True);subprocess.run(cmd,check=True)

def prepare():
    c=np.load(SRC/'calibration.npz');frames=sorted(p for p in SRC.iterdir() if (p/'rgb').exists());meta=[]
    for side in ['left','right']:
        out=OUT/'images'/side;out.mkdir(parents=True,exist_ok=True)
        K=c['mtx_'+side];maps=cv2.initUndistortRectifyMap(K,c['dist_'+side],None,K,(1440,1080),cv2.CV_32FC1)
        for i in range(20,46):
            im=cv2.imread(str(frames[i]/'rgb'/f'{side}_distorted.png'));im=cv2.remap(im,*maps,cv2.INTER_LINEAR)
            cv2.imwrite(str(out/f'{i:06d}.png'),im);meta.append(dict(frame=i,eye=side,image=f'{side}/{i:06d}.png'))
    (OUT/'input_manifest.json').write_text(json.dumps(dict(images=meta,raw_resolution=[1440,1080],intrinsics='fixed supplied per-eye calibration, images explicitly undistorted',validation_captures=[20,26,32,38,44],note='SFM uses all 52 photos; appearance training will exclude both eyes of each validation capture.'),indent=2))

def features():
    db=OUT/'database.db';images=OUT/'images'
    run(['colmap','feature_extractor','--database_path',str(db),'--image_path',str(images),'--ImageReader.single_camera_per_folder','1',
         '--ImageReader.camera_model','PINHOLE','--FeatureExtraction.use_gpu','0','--FeatureExtraction.num_threads','4',
         '--FeatureExtraction.max_image_size','1440','--SiftExtraction.max_num_features','11000','--SiftExtraction.peak_threshold','0.006'])
    c=np.load(SRC/'calibration.npz')
    with pycolmap.Database.open(db) as d:
        assigned={im.camera_id:im.name.split('/')[0] for im in d.read_all_images()}
        for camera in d.read_all_cameras():
            side=assigned[camera.camera_id];k=c['mtx_'+side];camera.params=np.array([k[0,0],k[1,1],k[0,2],k[1,2]])
            d.update_camera(camera);print('FIXED_CAMERA',camera,side,flush=True)

def match():
    run(['colmap','exhaustive_matcher','--database_path',str(OUT/'database.db'),'--FeatureMatching.use_gpu','0',
        '--FeatureMatching.num_threads','4','--SiftMatching.cpu_brute_force_matcher','0','--FeatureMatching.guided_matching','1'])

def mapping():
    (OUT/'sparse').mkdir(exist_ok=True)
    run(['colmap','mapper','--database_path',str(OUT/'database.db'),'--image_path',str(OUT/'images'),'--output_path',str(OUT/'sparse'),
        '--Mapper.num_threads','4','--Mapper.ba_refine_focal_length','0','--Mapper.ba_refine_principal_point','0','--Mapper.ba_refine_extra_params','0'])
    rows=[]
    for p in sorted((OUT/'sparse').iterdir()):
        if not p.is_dir():continue
        rec=pycolmap.Reconstruction(p);rows.append(dict(component=p.name,images=rec.num_reg_images(),points=rec.num_points3D(),
            observations=rec.compute_num_observations(),mean_reprojection_error=rec.compute_mean_reprojection_error(),registered=sorted(im.name for im in rec.images.values())))
    (OUT/'summary.json').write_text(json.dumps(rows,indent=2));print('SFM_SUMMARY',json.dumps(rows),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','features','match','mapping']);a=p.parse_args();globals()[a.stage]()
