#!/usr/bin/env python3
"""Separate LiDAR-assisted calibration of photo-predicted depths.

Fits only non-withheld scans, with spatial calibration-check anchors separated
inside each scan. This is not photograph-only reconstruction or independent GT.
"""
from pathlib import Path
import argparse,json
import numpy as np
from scipy.optimize import least_squares
ROOT=Path(__file__).resolve().parents[1];O=ROOT/'experiments/student_hall_geometry_20261031';B=ROOT/'experiments/student_hall_lidar'
def score(p,t):
    e=np.abs(p-t);return dict(median_cm=float(np.median(e)*100),p90_cm=float(np.percentile(e,90)*100),clipped_mae_cm=float(np.minimum(e,1).mean()*100))
def main(a):
    source=O/f'depth_{a.model}_{a.mode}_{a.width}';dest=O/f'depth_{a.model}cal_{a.mode}_{a.width}';dest.mkdir(exist_ok=True);c=dict(np.load(source/'calibration.npz'));K=c['K'];h,w=int(c['height']),int(c['width']);np.savez(dest/'calibration.npz',**c);rows=[]
    for path in sorted(source.glob('000*.npz')):
        i=int(path.stem);d=dict(np.load(path));pred=d['depth'];x=np.load(B/'frames'/path.name)['camera'];x=x[(np.linalg.norm(x,axis=1)>3)&(x[:,2]>1)&(x[:,2]<35)];uv=np.rint(x[:,:2]/x[:,2:]*[K[0,0],K[1,1]]+K[:2,2]).astype(int);good=(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h);x,uv=x[good],uv[good];order=np.argsort(x[:,2]);_,j=np.unique(uv[order,1]*w+uv[order,0],return_index=True);j=order[j];x,uv=x[j],uv[j]
        p=pred[uv[:,1],uv[:,0]];ok=p>0;p,t=p[ok],x[ok,2];u,v=uv[ok].T;train=(u//16+3*(v//16))%5!=0;check=~train;baseline=score(p[check],t[check]);candidates=[]
        for kind in ['affine','inverse']:
            def convert(p,q):return p*q[0]+q[1] if kind=='affine' else 1/np.maximum(q[0]/np.maximum(p,.01)+q[1],.005)
            init=[float(np.clip(np.median(t[train]/p[train]),.51,1.99)),0.] if kind=='affine' else [float(np.clip(np.median(p[train]/t[train]),.51,1.99)),0.]
            bounds=([.5,-5.],[2.,5.]) if kind=='affine' else ([.5,-.06],[2.,.06])
            fit=least_squares(lambda q:convert(p[train],q)-t[train],init,bounds=bounds,loss='soft_l1',f_scale=.15,max_nfev=150);s=score(convert(p[check],fit.x),t[check]);candidates.append(dict(kind=kind,parameters=fit.x.tolist(),check=s))
        best=min(candidates,key=lambda r:r['check']['clipped_mae_cm']);selected=best['check']['clipped_mae_cm']<baseline['clipped_mae_cm']*.95 and best['check']['p90_cm']<=baseline['p90_cm']
        if selected:
            aa,bb=best['parameters'];out=pred*aa+bb if best['kind']=='affine' else 1/np.maximum(aa/np.maximum(pred,.01)+bb,.005);out=np.where((pred>0)&(out>1)&(out<35),out,0).astype(np.float32)
        else:out=pred.copy()
        sparse=np.zeros((h,w),np.float32);sparse[uv[:,1],uv[:,0]]=x[:,2]
        np.savez_compressed(dest/path.name,raw_depth=pred,depth=out,sparse=sparse,rgb=d['rgb'],confidence=d['confidence'])
        row=dict(frame=i,train_anchors=int(train.sum()),check_anchors=int(check.sum()),raw_check=baseline,candidates=candidates,correction_selected=selected,selected=best if selected else None,caveat='LiDAR-assisted calibration using only this non-withheld scan. Spatial anchor check is not an independent scene test. Whole withheld scans never enter this calibration. Original DA3 output is preserved.')
        pathout=dest/path.with_suffix('.json').name;pathout.write_text(json.dumps(row,indent=2));rows.append(row);print('DEPTH_CALIBRATION',json.dumps(row),flush=True)
    (dest/'summary.json').write_text(json.dumps(rows,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['validation','final'],default='validation');p.add_argument('--width',type=int,default=504);p.add_argument('--model',default='da3');main(p.parse_args())
