#!/usr/bin/env python3
"""Previously unscored right-eye diagnostics, without image/pose fitting."""
from pathlib import Path
import argparse,json,sys
import cv2,numpy as np,pycolmap
from PIL import Image
from hall_research_experiments import O,P,PRE,VAL,read,dump,metric
def raw_pose(im):
    t=np.eye(4);t[:3]=im.cam_from_world().matrix();return np.linalg.inv(t)@np.diag([1.,-1.,-1.,1.])
def prepare():
    meta=read(P/'input_principal_960/manifest.json');r=pycolmap.Reconstruction(P/'input_principal_960/sparse/0');big=pycolmap.Reconstruction(P/'sfm_ba_principal');names={im.name:im for im in big.images.values()}
    poses=np.array([raw_pose(im) for im in sorted(r.images.values(),key=lambda im:im.name)],dtype=np.float32)
    # Match the native loader's float32 accumulation and Y/Z OpenGL convention.
    mean=np.zeros(3,np.float32)
    for t in poses:mean+=t[:3,3]
    mean/=len(poses);scale=np.float32(1/np.max(np.abs(poses[:,:3,3]-mean)))
    normalized=poses.copy();normalized[:,:3,3]=(poses[:,:3,3]-mean)*scale
    extra=[]
    folder=O/'probe_inputs';folder.mkdir()
    for k,frame in enumerate(VAL):
        t=raw_pose(names[f'right/{frame:06d}.png']).astype(np.float32);t[:3,3]=(t[:3,3]-mean)*scale;extra.append(t)
        im=cv2.imread(str(PRE/'stereo_colmap/images/right'/f'{frame:06d}.png'));im=cv2.resize(im,(720,540),interpolation=cv2.INTER_AREA);cv2.imwrite(str(folder/f'right_{frame:06d}.png'),im)
    train=[i for i,row in enumerate(meta['images']) if row['split']=='train'];rightref=next(j for j,i in enumerate(train) if meta['images'][i]['eye']=='right')
    np.savez(folder/'poses.npz',right_poses=extra,normalized_train_poses=normalized[train],right_reference=rightref,mean=mean,scale=scale)
    dump(folder/'plan.json',dict(frames=VAL,eye='right',fit_exposure_or_pose=False,use_for_checkpoint_selection=False,meaning='Additional cross-eye diagnostic. Both eyes of these timestamps excluded from appearance training, but poses estimated with all SfM images. Not an external blind test.',normalization='Native loader camera mean/maxabs; source 47-camera dataset unchanged.'))
    print('RIGHT_PROBE_PREPARED',len(extra),flush=True)
def evaluate(a):
    manifest=read(O/a.name/'manifest.json');wide=manifest['engine']=='wide'
    if wide:sys.path.insert(0,str((O/'native_package').resolve()))
    from msplat import Dataset,GaussianTrainer,TrainingConfig,cleanup
    data=Path(manifest['input']);meta=read(data/'manifest.json');ds=Dataset(str(data),downscale_factor=1,eval_mode=True,test_every=meta['stride']);cfg=TrainingConfig(iterations=10000,sh_degree=3,num_downscales=0,warmup_length=100000000,bg_color=[0,0,0]);t=GaussianTrainer(ds,cfg);t.load_checkpoint(str(O/a.name/'best_checkpoint'))
    p=np.load(O/'probe_inputs/poses.npz');diff=max(float(np.max(np.abs(ds.camera_pose(k)-pose))) for k,pose in enumerate(p['normalized_train_poses']));assert diff<1e-5,diff
    rows=[]
    for frame,pose in zip(VAL,p['right_poses']):
        im=np.asarray(t.render_from_pose(np.array(pose,dtype=np.float32),ref_cam_idx=int(p['right_reference'])));im=cv2.resize(im,(720,540),interpolation=cv2.INTER_AREA);im=np.uint8(np.clip(im,0,1)*255+.5)
        Image.fromarray(im).save(O/a.name/f'right_{frame:06d}.png');target=np.asarray(Image.open(O/'probe_inputs'/f'right_{frame:06d}.png'));rows.append(dict(frame=frame,**metric(im/255,target/255)))
    dump(O/a.name/'right_probe_metrics.json',dict(psnr=float(np.mean([r['psnr'] for r in rows])),ssim=float(np.mean([r['ssim'] for r in rows])),views=rows,pose_reproduction_max_abs_error=diff,checkpoint_selection_used_these_images=False));print('RIGHT_PROBE',a.name,json.dumps(rows),flush=True);cleanup()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','evaluate']);p.add_argument('--name');a=p.parse_args();prepare() if a.stage=='prepare' else evaluate(a)
