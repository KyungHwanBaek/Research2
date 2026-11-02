#!/usr/bin/env python3
"""Full-route 96-photo SfM extension and intrinsics BA variants, no old writes."""
from pathlib import Path
import argparse,json,shutil,subprocess
import cv2,numpy as np,pycolmap
O=Path('experiments/student_hall_refinement_20261027');PREV=Path('experiments/student_hall_refinement_20261026');S=O/'sfm_full';RAW=Path('data/extracted/01-26-15-41-24')
def run(cmd):print('COMMAND',json.dumps(cmd),flush=True);subprocess.run(cmd,check=True)
def summarize(path):
    rec=pycolmap.Reconstruction(path)
    return dict(images=rec.num_reg_images(),points=rec.num_points3D(),observations=rec.compute_num_observations(),mean_error_px=rec.compute_mean_reprojection_error(),cameras=[dict(id=c.camera_id,model=c.model_name,params=c.params.tolist(),width=c.width,height=c.height) for c in rec.cameras.values()],registered=sorted(im.name for im in rec.images.values()))
def prepare():
    S.mkdir(parents=True,exist_ok=True);target=S/'database.db';assert not target.exists()
    # SQLite backup preserves camera/image/frame/feature IDs used by the
    # existing 52-view model, enabling safe continuation with extra images.
    import sqlite3
    src=sqlite3.connect('file:'+str(PREV/'stereo_colmap/database.db')+'?mode=ro',uri=True);dst=sqlite3.connect(target);src.backup(dst);dst.close();src.close()
    frames=sorted(p for p in RAW.iterdir() if (p/'rgb').exists());cal=np.load(RAW/'calibration.npz')
    cv2.setNumThreads(4)
    for side in ['left','right']:
        folder=S/'images'/side;folder.mkdir(parents=True,exist_ok=True);K=cal['mtx_'+side];maps=cv2.initUndistortRectifyMap(K,cal['dist_'+side],None,K,(1440,1080),cv2.CV_32FC1)
        for i,frame in enumerate(frames):
            dest=folder/f'{i:06d}.png';old=PREV/'stereo_colmap/images'/side/f'{i:06d}.png'
            if old.exists():shutil.copyfile(old,dest)
            else:cv2.imwrite(str(dest),cv2.remap(cv2.imread(str(frame/'rgb'/f'{side}_distorted.png')),*maps,cv2.INTER_LINEAR))
    print('PREPARED',96,'images; copied database and reused 52 cached images',flush=True)
def features():
    run(['colmap','feature_extractor','--database_path',str(S/'database.db'),'--image_path',str(S/'images'),'--ImageReader.single_camera_per_folder','1','--ImageReader.camera_model','PINHOLE','--FeatureExtraction.use_gpu','0','--FeatureExtraction.num_threads','4','--FeatureExtraction.max_image_size','1440','--SiftExtraction.max_num_features','11000','--SiftExtraction.peak_threshold','0.006'])
    c=np.load(RAW/'calibration.npz')
    with pycolmap.Database.open(S/'database.db') as d:
        assign={im.camera_id:im.name.split('/')[0] for im in d.read_all_images()}
        for cam in d.read_all_cameras():
            k=c['mtx_'+assign[cam.camera_id]];cam.params=np.array([k[0,0],k[1,1],k[0,2],k[1,2]]);d.update_camera(cam)
    print('CAMERAS fixed per eye',flush=True)
def match():run(['colmap','exhaustive_matcher','--database_path',str(S/'database.db'),'--FeatureMatching.use_gpu','0','--FeatureMatching.num_threads','4','--SiftMatching.cpu_brute_force_matcher','0','--FeatureMatching.guided_matching','1'])
def mapping():
    (S/'sparse').mkdir(exist_ok=True)
    run(['colmap','mapper','--database_path',str(S/'database.db'),'--image_path',str(S/'images'),'--input_path',str(PREV/'stereo_colmap/sparse/0'),'--output_path',str(S/'sparse'),'--Mapper.num_threads','4','--Mapper.ba_refine_focal_length','0','--Mapper.ba_refine_principal_point','0','--Mapper.ba_refine_extra_params','0'])
    rows=[]
    for p in sorted((S/'sparse').iterdir()):
        if p.is_dir() and (p/'cameras.bin').exists():rows.append(dict(component=p.name,**summarize(p)))
    # Some COLMAP builds write continuation straight into output_path.
    if (S/'sparse/cameras.bin').exists():rows.append(dict(component='.',**summarize(S/'sparse')))
    (S/'summary.json').write_text(json.dumps(rows,indent=2));print('FULL_SFM',json.dumps(rows),flush=True)
def ba(mode):
    out=O/f'sfm_ba_{mode}';out.mkdir(exist_ok=True)
    run(['colmap','bundle_adjuster','--input_path',str(PREV/'stereo_colmap/sparse/0'),'--output_path',str(out),'--BundleAdjustment.refine_focal_length','1','--BundleAdjustment.refine_principal_point','1' if mode=='principal' else '0','--BundleAdjustment.refine_extra_params','0','--BundleAdjustmentCeres.max_num_iterations','150'])
    row=summarize(out);(out/'summary.json').write_text(json.dumps(row,indent=2));print('BA_VARIANT',mode,json.dumps(row),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','features','match','mapping','ba_focal','ba_principal']);a=p.parse_args();ba(a.stage[3:]) if a.stage.startswith('ba_') else globals()[a.stage]()
