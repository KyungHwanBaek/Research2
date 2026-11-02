#!/usr/bin/env python3
"""Global multi-scan surface refinement; even returns fit, odd returns check."""
from pathlib import Path
import json,time
import numpy as np
import open3d as o
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

B=Path('experiments/student_hall_lidar');OUT=Path('experiments/student_hall_refinement_20261026')
def pc(x,voxel,radius=.45):
    p=o.geometry.PointCloud(o.utility.Vector3dVector(x.astype(float))).voxel_down_sample(voxel)
    p.estimate_normals(o.geometry.KDTreeSearchParamHybrid(radius=radius,max_nn=30));return p
def trans(x,P):return x@P[:3,:3].T+P[:3,3]

def evaluate(P,clouds):
    common=set(np.load(B/'colmap_metric_poses.npz')['indices'].tolist());pairs=[(i,i+1) for i in range(47) if i in common and i+1 in common]
    pts=[];norm=[];trees=[];src=[]
    for x in clouds:
        p=pc(x[::2],.12,radius=.5);p2=pc(x[1::2],.18);pts.append(np.asarray(p.points));norm.append(np.asarray(p.normals));trees.append(cKDTree(pts[-1]));src.append(np.asarray(p2.points))
    def score(poses):
        ds=[];planes=[]
        for a,b in pairs:
            x=trans(src[a],np.linalg.inv(poses[b])@poses[a]);d,ix=trees[b].query(x,workers=4);pl=np.abs(np.sum((x-pts[b][ix])*norm[b][ix],axis=1));ds.extend(d);planes.extend(pl)
        ds=np.array(ds);planes=np.array(planes)
        return dict(overlap_25cm=float(np.mean(ds<.25)),matched_plane_median_cm=float(np.median(planes[ds<.25])*100),clipped_plane_mean_cm=float(np.minimum(planes,.5).mean()*100),samples=len(ds))
    base=np.load(B/'posegraph_poses.npz')['poses'];return dict(baseline=score(base),global_refinement=score(P),caveat='Identical 37 adjacent pairs and odd returns excluded from fitting. Target even returns and poses reused, so consistency only, not survey ground truth.')

def main():
    P=np.load(B/'posegraph_poses.npz')['poses'].copy();Pbase=P.copy();rng=np.random.default_rng(42);xx=[];nn=[];trees=[];raw=[]
    for i in range(48):
        x=np.load(B/'frames'/f'{i:06d}.npz')['camera'];raw.append(x);x=x[::2];x=x[np.linalg.norm(x,axis=1)>3]
        p=pc(x,.15);xx.append(np.asarray(p.points));nn.append(np.asarray(p.normals));trees.append(cKDTree(xx[-1]))
    pairs=json.loads((B/'posegraph.json').read_text())['edges'];history=[]
    for outer in range(4):
        aa=[];bb=[];xa=[];xb=[];norm=[]
        for a,b in pairs:
            # Symmetric directed matches prevent a single dense scan dominating.
            for a,b in [(a,b),(b,a)]:
                T=np.linalg.inv(P[b])@P[a];src=trans(xx[a],T);d,j=trees[b].query(src,workers=4)
                n=nn[a]@T[:3,:3].T;consistent=np.abs(np.sum(n*nn[b][j],axis=1))>.9
                ids=np.flatnonzero((d<.20)&consistent)
                if len(ids)>450:ids=rng.choice(ids,450,replace=False)
                aa.extend([a]*len(ids));bb.extend([b]*len(ids));xa.extend(xx[a][ids]);xb.extend(xx[b][j[ids]]);norm.extend(nn[b][j[ids]])
        aa=np.array(aa);bb=np.array(bb);xa=np.array(xa);xb=np.array(xb);norm=np.array(norm)
        xbase=np.r_[Rotation.from_matrix(P[:,:3,:3]).as_rotvec().ravel(),P[:,:3,3].ravel()]
        def matrices(v):
            dv=v.reshape(48,6);R=Rotation.from_rotvec(dv[:,:3]).as_matrix()@P[:,:3,:3];t=P[:,:3,3]+dv[:,3:];return R,t
        def residual(v):
            R,t=matrices(v);wa=np.einsum('nij,nj->ni',R[aa],xa)+t[aa];wb=np.einsum('nij,nj->ni',R[bb],xb)+t[bb]
            n=np.einsum('nij,nj->ni',R[bb],norm);plane=np.sum((wa-wb)*n,axis=1)/.025
            rot=Rotation.from_matrix(R@Pbase[:,:3,:3].transpose(0,2,1)).as_rotvec()/.006;tr=(t-Pbase[:,:3,3])/.12
            # Strong fixed global gauge at frame zero, no absolute-world claim.
            rot[0]*=100;tr[0]*=100
            return np.r_[plane,rot.ravel(),tr.ravel()]
        n=len(aa);sp=lil_matrix((n+288,288),dtype=np.int8)
        for k,(a,b) in enumerate(zip(aa,bb)):sp[k,a*6:a*6+6]=1;sp[k,b*6:b*6+6]=1
        for i in range(48):sp[n+i*3:n+i*3+3,i*6:i*6+6]=1;sp[n+144+i*3:n+144+i*3+3,i*6:i*6+6]=1
        t0=time.time();fit=least_squares(residual,np.zeros(288),jac_sparsity=sp.tocsr(),loss='soft_l1',f_scale=1,x_scale='jac',max_nfev=80,ftol=1e-4)
        R,t=matrices(fit.x);P[:,:3,:3]=R;P[:,:3,3]=t
        row=dict(outer=outer,constraints=n,nfev=fit.nfev,cost=float(fit.cost),seconds=time.time()-t0,
            maximum_translation_change_cm=float(np.max(np.linalg.norm(P[:,:3,3]-Pbase[:,:3,3],axis=1))*100));history.append(row);print('global_scan_fit',json.dumps(row),flush=True)
    np.savez(OUT/'global_poses.npz',poses=P,indices=np.arange(48));report=evaluate(P,raw);report['fit_history']=history
    (OUT/'global_pose_evaluation.json').write_text(json.dumps(report,indent=2));print('pose_evaluation',json.dumps(report),flush=True)
if __name__=='__main__':main()
