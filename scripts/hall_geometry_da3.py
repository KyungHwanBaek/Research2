#!/usr/bin/env python3
"""Pose-conditioned multiview photo depths using official DA3-Small weights.

Uses the official network directly, ImageNet normalization and official pose
normalization/scale alignment. Avoids optional CUDA rendering dependencies.
"""
from pathlib import Path
import argparse,json,sys,time
import cv2,numpy as np,torch
from omegaconf import OmegaConf
from safetensors.torch import load_file
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'third_party/depth_anything_3/src'))
from depth_anything_3.cfg import create_object
from depth_anything_3.utils.pose_align import align_poses_umeyama
OUT=ROOT/'experiments/student_hall_geometry_20261031';B=ROOT/'experiments/student_hall_lidar';HOLD=[5,13,21,29,37,45]
def main(a):
    torch.set_num_threads(4);cv2.setNumThreads(2);device='mps' if torch.backends.mps.is_available() else 'cpu';weights=ROOT/f'third_party/depth_anything_3/weights/DA3-{a.model.upper()}';config=json.loads((weights/'config.json').read_text());model=create_object(OmegaConf.create(config['config']));state={k.removeprefix('model.'):v for k,v in load_file(weights/'model.safetensors').items()}
    load=model.load_state_dict(state,strict=False)
    # Current upstream adds auxiliary ray heads absent from the released
    # checkpoint. Main depth/confidence and camera-decoder weights all load.
    # Ray-based pose decoding is disabled; these outputs are discarded upstream.
    assert not load.unexpected_keys and all('output_conv2_aux' in k for k in load.missing_keys),str(load)
    print('CHECKPOINT_LOAD',str(load),flush=True);model.eval().to(device)
    w=a.width;h=round((w*.75)/14)*14;cal=np.load(B/'fused/selected_poses.npz');P,K=cal['poses'],cal['K'].copy();K[0,:]*=w/960;K[1,:]*=h/720
    K[0,2]+=.5*w/960-.5;K[1,2]+=.5*h/720-.5
    label='da3' if a.model=='small' else 'da3large';folder=OUT/f'depth_{label}_{a.mode}_{w}';folder.mkdir(exist_ok=True);np.savez(folder/'calibration.npz',K=K,poses=P,width=w,height=h)
    ids=[i for i in range(20,46) if a.mode=='final' or i not in HOLD];rows=[];batches=[ids[j:j+a.batch] for j in range(0,len(ids),a.batch)]
    for batch in batches:
        if len(batch)<3:batch=ids[-a.batch:]
        if all((folder/f'{i:06d}.json').exists() for i in batch):
            rows.extend(json.loads((folder/f'{i:06d}.json').read_text()) for i in batch);continue
        start=time.time();images=np.array([cv2.resize(cv2.imread(str(B/'images'/f'{i:06d}.png'))[:,:,::-1],(w,h),interpolation=cv2.INTER_AREA) for i in batch]);x=torch.from_numpy(images.copy()).permute(0,3,1,2).float()/255;x=(x-torch.tensor([.485,.456,.406])[None,:,None,None])/torch.tensor([.229,.224,.225])[None,:,None,None]
        ex=np.linalg.inv(P[batch]).astype(np.float32);ext=ex@P[batch[0]].astype(np.float32);centers=np.linalg.inv(ext)[:,:3,3];median=max(float(np.median(np.linalg.norm(centers,axis=1))),.1);ext[:,:3,3]/=median
        with torch.inference_mode():
            pred=model(x[None].to(device),torch.from_numpy(ext[None]).to(device),torch.from_numpy(np.repeat(K[None,None],len(batch),axis=1).astype(np.float32)).to(device),use_ray_pose=False,ref_view_strategy='first')
        extpred=pred.extrinsics[0].float().cpu().numpy();_,_,scale=align_poses_umeyama(extpred,ex,ransac=False)
        depth=pred.depth[0].float().cpu().numpy()/scale;confidence=pred.depth_conf[0].float().cpu().numpy()
        for j,i in enumerate(batch):
            d=depth[j];valid=(d>1)&(d<35)&np.isfinite(d);conf=confidence[j];cut=np.percentile(conf[valid],30) if valid.any() else 0;filtered=np.where(valid&(conf>=cut),d,0).astype(np.float32)
            # Raw predicted depth remains separate from the confidence mask.
            np.savez_compressed(folder/f'{i:06d}.npz',raw_depth=filtered,depth=filtered,unmasked_depth=d,confidence=conf,rgb=images[j])
            row=dict(frame=i,batch=batch,mode=a.mode,scale_alignment=float(scale),network_pose_normalization_m=median,valid_fraction=float(np.mean(filtered>0)),confidence_cut=float(cut),device=device,batch_seconds=time.time()-start,note='Photo-derived DA3 depth, conditioned on existing LiDAR-derived camera poses. No LiDAR depth fitting. Whole validation captures excluded from all inference batches in validation mode. Learned geometry is a prediction and not a sensor measurement.')
            (folder/f'{i:06d}.json').write_text(json.dumps(row,indent=2));rows.append(row)
        print('DA3_BATCH',json.dumps(dict(batch=batch,seconds=time.time()-start,scale=float(scale),depth_min_max=[float(np.min(depth)),float(np.max(depth))])),flush=True)
        del pred
        if device=='mps':torch.mps.empty_cache()
    (folder/'summary.json').write_text(json.dumps(rows,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['validation','final'],default='validation');p.add_argument('--width',type=int,default=504);p.add_argument('--batch',type=int,default=6);p.add_argument('--model',choices=['small','large'],default='small');main(p.parse_args())
