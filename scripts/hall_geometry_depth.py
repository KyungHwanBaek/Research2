#!/usr/bin/env python3
"""Metric photo stereo inference: official POSTECH RGB-NIR and RAFT-Stereo.

No weights are trained on these captures. Raw predictions are preserved;
LiDAR is used for diagnostics and an explicitly separate disparity correction.
The public POSTECH pretrained model may have seen this dataset during training;
therefore these diagnostics do not establish unseen-dataset generalization.
"""
from pathlib import Path
import argparse,json,sys,time,warnings
from types import SimpleNamespace
import cv2,numpy as np,torch
import torch.nn.functional as F
from scipy.optimize import least_squares
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/student_hall_geometry_20261031'
RAW=ROOT/'data/extracted/01-26-15-41-24';BASE=ROOT/'experiments/student_hall_lidar'
FRAMES=sorted(p for p in RAW.iterdir() if (p/'rgb').is_dir())
def calibration(width):
    c=dict(np.load(RAW/'calibration.npz'));s=np.load(ROOT/'experiments/student_hall_refinement_20261026/stereo_calibration.npz');height=width*3//4
    R1,R2,P1,P2,Q,_,_=cv2.stereoRectify(c['mtx_left'],c['dist_left'],c['mtx_right'],c['dist_right'],(1440,1080),s['R'],s['T'].reshape(3,1),newImageSize=(width,height),alpha=0,flags=cv2.CALIB_ZERO_DISPARITY)
    maps=[cv2.initUndistortRectifyMap(c['mtx_'+side],c['dist_'+side],r,p[:3,:3],(width,height),cv2.CV_32FC1) for side,r,p in [('left',R1,P1),('right',R2,P2)]]
    P=np.load(BASE/'fused/selected_poses.npz')['poses'];T=np.eye(4);T[:3,:3]=R1.T
    return dict(K=P1[:3,:3],fB=abs(P2[0,3]),R1=R1,poses=P@T,width=width,height=height),maps
def sparse_depth(i,cal):
    x=np.load(BASE/'frames'/f'{i:06d}.npz')['camera'];x=x[np.linalg.norm(x,axis=1)>3]@cal['R1'].T
    good=(x[:,2]>1)&(x[:,2]<35);x=x[good];uv=np.rint(x[:,:2]/x[:,2:]*[cal['K'][0,0],cal['K'][1,1]]+cal['K'][:2,2]).astype(int)
    good=(uv[:,0]>=0)&(uv[:,0]<cal['width'])&(uv[:,1]>=0)&(uv[:,1]<cal['height']);uv=uv[good];x=x[good]
    order=np.argsort(x[:,2]);_,j=np.unique(uv[order,1]*cal['width']+uv[order,0],return_index=True);j=order[j]
    d=np.zeros((cal['height'],cal['width']),np.float32);d[uv[j,1],uv[j,0]]=x[j,2];return d
def load_model(kind,device):
    if kind=='rgbnir':
        sys.path.insert(0,str(ROOT/'third_party/pixel_aligned_rgb_nir'))
        from arguments import Arguments
        from net.feature_fusion import RAFTStereoFusion
        args=Arguments();args.shared_fusion=True;args.shared_backbone=False;args.corr_implementation='reg';args.n_gru_layers=3;args.n_downsample=2;args.mixed_precision=False
        model=RAFTStereoFusion(args);weights=ROOT/'third_party/pixel_aligned_rgb_nir/weights/model_feature_fusion.pth'
        state=torch.load(weights,map_location='cpu',weights_only=True)['model_state_dict']
    else:
        sys.path.insert(0,str(ROOT/'third_party/raft_stereo'))
        from core.raft_stereo import RAFTStereo
        args=SimpleNamespace(hidden_dims=[128]*3,corr_levels=4,corr_radius=4,n_downsample=2,context_norm='batch',n_gru_layers=3,shared_backbone=False,mixed_precision=False,corr_implementation='reg',slow_fast_gru=False)
        model=RAFTStereo(args);weights=ROOT/'third_party/raft_stereo/raftstereo-middlebury.pth'
        state={k.removeprefix('module.'):v for k,v in torch.load(weights,map_location='cpu',weights_only=True).items()}
    model.load_state_dict(state,strict=True);model.eval().to(device);return model,weights
def stats(pred,target,mask=None):
    anchors=target>0
    if mask is not None:anchors &= mask
    good=anchors&(pred>0)&np.isfinite(pred);e=np.abs(pred[good]-target[good]);rel=e/target[good]
    return dict(anchors=int(anchors.sum()),compared=int(good.sum()),coverage=float(good.sum()/max(anchors.sum(),1)),median_error_cm=float(np.median(e)*100) if len(e) else None,p90_error_cm=float(np.percentile(e,90)*100) if len(e) else None,clipped_mae_cm=float(np.minimum(e,1).mean()*100) if len(e) else None,fraction_within_10pct=float(np.mean(rel<.1)) if len(e) else None)
def main(a):
    torch.set_num_threads(4);cv2.setNumThreads(3);device=torch.device(a.device if a.device!='auto' else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    model,weights=load_model(a.model,device);cal,maps=calibration(a.width);folder=OUT/f'depth_{a.model}_{a.width}';folder.mkdir(parents=True,exist_ok=True);np.savez(folder/'calibration.npz',**cal)
    rows=[];ids=[int(x) for x in a.frames.split(',')] if a.frames else list(range(20,46))
    for i in ids:
        dest=folder/f'{i:06d}.npz'
        if dest.exists():rows.append(json.loads(dest.with_suffix('.json').read_text()));continue
        start=time.time();images=[]
        for mod,side,j in [('rgb','left',0),('rgb','right',1),('nir','left',0),('nir','right',1)]:
            im=cv2.imread(str(FRAMES[i]/mod/f'{side}_distorted.png'));im=cv2.remap(im,*maps[j],cv2.INTER_LINEAR)[:,:,::-1].copy();images.append(im)
        tensors=[torch.from_numpy(im).permute(2,0,1)[None].float().to(device) for im in images]
        h,w=cal['height'],cal['width'];ph=(-h)%32;pw=(-w)%32
        tensors=[F.pad(x,(0,pw,0,ph),mode='replicate') for x in tensors]
        if a.model=='nir':tensors=tensors[2:]
        elif a.model!='rgbnir':tensors=tensors[:2]
        def infer(xs):
            with torch.inference_mode(),warnings.catch_warnings():
                warnings.simplefilter('ignore');result=model(*xs,iters=a.iters,test_mode=True)[1]
                return -result[0,0,:h,:w].float().cpu().numpy()
        dl=infer(tensors)
        # Swap & horizontally mirror both modalities; unpad happens after mirror
        # to keep matching coordinates consistent for a non-multiple-of-32 height.
        reverse=[]
        for j in range(0,len(tensors),2):
            reverse.extend([F.pad(tensors[j+1][...,:h,:w].flip(-1),(0,pw,0,ph),mode='replicate'),F.pad(tensors[j][...,:h,:w].flip(-1),(0,pw,0,ph),mode='replicate')])
        dr=infer(reverse)[:,::-1].copy();yy,xx=np.indices(dl.shape);xr=xx-dl
        sampled=cv2.remap(dr,xr.astype(np.float32),yy.astype(np.float32),cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=-999)
        lr_error=np.abs(dl-sampled);valid=(dl>1)&(xr>=0)&(xr<w)&(lr_error<1.)
        raw=cal['fB']/np.maximum(dl,.001);valid&=(raw>1)&(raw<35);depth=np.where(valid,raw,0).astype(np.float32)
        sparse=sparse_depth(i,cal);train=((xx//32+3*(yy//32))%5)!=0;anchor=(sparse>0)&valid
        # Separate, bounded correction; raw network output is always retained.
        fitmask=anchor&train;target=cal['fB']/np.maximum(sparse[fitmask],1e-6);disp=dl[fitmask]
        correction=[1.,0.];selected=False
        if fitmask.sum()>100:
            fit=least_squares(lambda v:v[0]*disp+v[1]-target,[1.,0.],bounds=([.85,-1.5],[1.15,1.5]),loss='soft_l1',f_scale=.25)
            correction=fit.x.tolist();dcal=cal['fB']/np.maximum(dl*fit.x[0]+fit.x[1],.01);dcal=np.where(valid&(dcal>1)&(dcal<35),dcal,0).astype(np.float32)
            old=stats(depth,sparse,~train);new=stats(dcal,sparse,~train)
            selected=new['clipped_mae_cm'] is not None and new['clipped_mae_cm']<old['clipped_mae_cm']*.98
            corrected=dcal if selected else depth
        else:corrected=depth.copy()
        row=dict(frame=i,model=a.model,width=w,iters=a.iters,device=str(device),seconds=time.time()-start,valid_fraction=float(valid.mean()),raw_anchor=stats(depth,sparse),anchor_check_raw=stats(depth,sparse,~train),anchor_check_corrected=stats(corrected,sparse,~train),disparity_correction=correction,correction_selected=selected,checkpoint=str(weights.relative_to(ROOT)),note='Depth from physical calibrated stereo baseline. Correction, if selected, uses spatially split LiDAR anchors from this same scan and is not an independent test. Public RGB-NIR checkpoint training overlap with source sequence is not excluded.')
        np.savez_compressed(dest,disparity=dl.astype(np.float32),lr_error=lr_error.astype(np.float32),raw_depth=depth,depth=corrected,sparse=sparse,rgb=images[0]);dest.with_suffix('.json').write_text(json.dumps(row,indent=2));rows.append(row);print('DEPTH',json.dumps(row),flush=True)
        color=cv2.applyColorMap(np.uint8(np.clip(corrected/25,0,1)*255),cv2.COLORMAP_TURBO);color[corrected==0]=0
        if i in [22,26,34,40,44]:cv2.imwrite(str(folder/f'preview_{i:02d}.jpg'),np.hstack((images[0][:,:,::-1],color)))
        if device.type=='mps':torch.mps.empty_cache()
    (folder/'last_batch.json').write_text(json.dumps(rows,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['rgbnir','rgb','nir'],default='rgbnir');p.add_argument('--width',type=int,default=960);p.add_argument('--iters',type=int,default=12);p.add_argument('--frames');p.add_argument('--device',default='auto');main(p.parse_args())
