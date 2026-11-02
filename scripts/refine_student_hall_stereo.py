#!/usr/bin/env python3
"""Recorded stereo extension; raw distorted images, fixed metric baseline.

No old experiment files are changed. Stereo calibration is fitted on one
subset and checked on disjoint image pairs; six complete LiDAR scans are
reserved for mesh validation (camera poses still use all scans).
"""
from pathlib import Path
import argparse,json,time
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from reconstruct_student_hall_lidar import dense_depth,transform

SRC=Path('data/extracted/01-26-15-41-24')
BASE=Path('experiments/student_hall_lidar')
OUT=Path('experiments/student_hall_refinement_20261026')
FRAMES=sorted(p for p in SRC.iterdir() if (p/'rgb').is_dir())
TRAIN=[1,6,10,16,22,28,34,40,46]
CHECK=[3,9,18,25,32,39,43,47]
HOLD=[5,13,21,29,37,45]

def dump(name,obj):
    (OUT/name).parent.mkdir(parents=True,exist_ok=True)
    (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2))

def correspondences(i,c,mod='rgb'):
    sift=cv2.SIFT_create(nfeatures=6500,contrastThreshold=.015)
    ims=[cv2.imread(str(FRAMES[i]/mod/f'{s}_distorted.png'),0) for s in ['left','right']]
    kd=[sift.detectAndCompute(im,None) for im in ims]
    matches=cv2.BFMatcher().knnMatch(kd[0][1],kd[1][1],k=2)
    ms=[m for m,n in matches if m.distance<.66*n.distance]
    a=np.float32([kd[0][0][m.queryIdx].pt for m in ms]);b=np.float32([kd[1][0][m.trainIdx].pt for m in ms])
    a=cv2.undistortPoints(a[:,None],c['mtx_left'],c['dist_left'])[:,0]
    b=cv2.undistortPoints(b[:,None],c['mtx_right'],c['dist_right'])[:,0]
    _,mask=cv2.findFundamentalMat(a*1800,b*1800,cv2.USAC_MAGSAC,1.2,.999,10000)
    ok=mask.ravel()>0
    return a[ok],b[ok]

def rectification(c,R,T):
    r1,r2,p1,p2,Q,_,_=cv2.stereoRectify(c['mtx_left'],c['dist_left'],c['mtx_right'],c['dist_right'],
        (1440,1080),R,np.ascontiguousarray(T).reshape(3,1),newImageSize=(960,720),alpha=0,flags=cv2.CALIB_ZERO_DISPARITY)
    maps=[cv2.initUndistortRectifyMap(c[f'mtx_{s}'],c[f'dist_{s}'],r,p[:3,:3],(960,720),cv2.CV_32FC1)
          for s,r,p in [('left',r1,p1),('right',r2,p2)]]
    return r1,r2,p1,p2,maps

def rect_errors(a,b,R,T,c):
    r1,r2,p1,p2,_=rectification(c,R,T)
    aa=cv2.perspectiveTransform(a[:,None],p1[:3,:3]@r1)[:,0]
    bb=cv2.perspectiveTransform(b[:,None],p2[:3,:3]@r2)[:,0]
    return np.abs(aa[:,1]-bb[:,1])

def calibrate():
    cv2.setNumThreads(4)
    c=dict(np.load(SRC/'calibration.npz'));matches={};rows=[]
    (OUT/'stereo_matches').mkdir(exist_ok=True)
    for i in TRAIN+CHECK:
        path=OUT/'stereo_matches'/f'{i:06d}.npz'
        if path.exists():
            z=np.load(path);a,b=z['a'],z['b']
        else:
            a,b=correspondences(i,c);np.savez(path,a=a,b=b)
        matches[i]=(a,b)
        print('stereo_features',i,len(a),flush=True)
    a=np.concatenate([matches[i][0] for i in TRAIN]);b=np.concatenate([matches[i][1] for i in TRAIN])
    ah=np.column_stack((a,np.ones(len(a))));bh=np.column_stack((b,np.ones(len(b))))
    R0=c['R'];T0=c['T'].ravel()/1000;baseline=np.linalg.norm(T0)
    ax=[];br=[]
    K=np.load(BASE/'calibration.npz')['K']
    for i in TRAIN:
        aa,bb=matches[i];uv=np.rint(aa*[K[0,0],K[1,1]]+K[:2,2]).astype(int)
        inside=(uv[:,0]>=0)&(uv[:,0]<960)&(uv[:,1]>=0)&(uv[:,1]<720)
        dep=np.load(BASE/'frames'/f'{i:06d}.npz')['depth'];d=np.zeros(len(aa));d[inside]=dep[uv[inside,1],uv[inside,0]]
        ok=(d>3)&(d<30);ax.append(np.column_stack((aa[ok],np.ones(ok.sum())))*d[ok,None]);br.append(bb[ok])
    ax=np.concatenate(ax);br=np.concatenate(br)
    def unpack(x):
        R=Rotation.from_rotvec(x[:3]).as_matrix()@R0;T=T0+np.r_[0,x[3:]];T=T/np.linalg.norm(T)*baseline
        return R,T
    def residual(x):
        R,T=unpack(x);E=np.array([[0,-T[2],T[1]],[T[2],0,-T[0]],[-T[1],T[0],0]])@R
        ea=ah@E.T;eb=bh@E
        epi=(np.sum(bh*ea,axis=1)/np.sqrt(ea[:,0]**2+ea[:,1]**2+eb[:,0]**2+eb[:,1]**2))*1800
        xr=ax@R.T+T;metric=(xr[:,:2]/xr[:,2:]-br)*1800
        return np.r_[epi,.7*metric.ravel()]
    # The unconstrained essential fit can reverse translation and collapse
    # metric scale on distant structure. Use known LiDAR depths plus small,
    # physical corrections around the supplied fixed camera rig instead.
    fit=least_squares(residual,np.zeros(5),bounds=([-0.004]*3+[-.008]*2,[.004]*3+[.008]*2),loss='soft_l1',f_scale=.6,max_nfev=150)
    R,T=unpack(fit.x)
    for i,(aa,bb) in matches.items():
        e0=rect_errors(aa,bb,R0,T0,c);e1=rect_errors(aa,bb,R,T,c)
        rows.append(dict(frame=i,split='fit' if i in TRAIN else 'check',matches=len(aa),
            original_median_px=float(np.median(e0)),refined_median_px=float(np.median(e1)),
            original_p90_px=float(np.percentile(e0,90)),refined_p90_px=float(np.percentile(e1,90))))
    e0=np.concatenate([rect_errors(*matches[i],R0,T0,c) for i in CHECK]);e1=np.concatenate([rect_errors(*matches[i],R,T,c) for i in CHECK])
    select=np.median(e1)<np.median(e0)*.9 and np.percentile(e1,90)<np.percentile(e0,90)
    Rs,Ts=(R,T) if select else (R0,T0)
    np.savez(OUT/'stereo_calibration.npz',R=Rs,T=Ts,R_original=R0,T_original=T0,R_refined=R,T_refined=T)
    report=dict(baseline_m=float(baseline),selected='refined' if select else 'original',
        check_median_px_original=float(np.median(e0)),check_median_px_refined=float(np.median(e1)),
        rotation_change_deg=float(np.linalg.norm(Rotation.from_matrix(R@R0.T).as_rotvec())*180/np.pi),
        translation_direction_change_deg=float(np.arccos(np.clip(T@T0/baseline**2,-1,1))*180/np.pi),views=rows,
        metric_training_matches=len(ax),caveat='Fixed baseline, LiDAR-supported right-image reprojection in fit; calibration-check pairs are disjoint. Depth validation still required.')
    dump('stereo_calibration.json',report);print(json.dumps(report),flush=True)

def matcher():
    return cv2.StereoSGBM_create(minDisparity=0,numDisparities=160,blockSize=5,P1=8*25,P2=32*25,
        disp12MaxDiff=1,preFilterCap=31,uniquenessRatio=12,speckleWindowSize=80,speckleRange=2,mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

def stereo(left,right,fB):
    # Matching the reversed, horizontally flipped pair gives positive disparity
    # in the right view, using exactly the same consistency filter.
    m=matcher();dl=m.compute(left,right).astype(np.float32)/16
    dr=m.compute(np.ascontiguousarray(right[:,::-1]),np.ascontiguousarray(left[:,::-1])).astype(np.float32)[:,::-1]/16
    yy,xx=np.indices(dl.shape);xr=np.rint(xx-dl).astype(int);inside=(xr>=0)&(xr<dl.shape[1])
    reverse=dr[yy,np.clip(xr,0,dl.shape[1]-1)]
    texture=cv2.boxFilter(left.astype(np.float32)**2,-1,(7,7))-cv2.boxFilter(left.astype(np.float32),-1,(7,7))**2
    z=fB/np.maximum(dl,.001)
    good=inside&(np.abs(dl-reverse)<.8)&(dl>2)&(dl<159)&(texture>12)&(z>1)&(z<35)
    out=np.where(good,z,0).astype(np.float32)
    return out

def depths():
    cv2.setNumThreads(4)
    c=dict(np.load(SRC/'calibration.npz'));s=np.load(OUT/'stereo_calibration.npz')
    r1,r2,p1,p2,maps=rectification(c,s['R'],s['T']);K=p1[:3,:3];fB=abs(p2[0,3])
    pp=np.load(BASE/'fused/selected_poses.npz')['poses'];A=np.eye(4);A[:3,:3]=r1.T
    np.savez(OUT/'rectified_poses.npz',poses=pp@A,K=K,R1=r1,shape=[720,960])
    (OUT/'stereo').mkdir(exist_ok=True);rows=[]
    for i,frame in enumerate(FRAMES):
        dest=OUT/'stereo'/f'{i:06d}.npz'
        if dest.exists():
            rows.append(json.loads(dest.with_suffix('.json').read_text()));continue
        t=time.time();rgb=[cv2.remap(cv2.imread(str(frame/'rgb'/f'{side}_distorted.png')),*maps[j],cv2.INTER_LINEAR) for j,side in enumerate(['left','right'])]
        nir=[cv2.remap(cv2.imread(str(frame/'nir'/f'{side}_distorted.png'),0),*maps[j],cv2.INTER_LINEAR) for j,side in enumerate(['left','right'])]
        dg=stereo(*[cv2.cvtColor(im,cv2.COLOR_BGR2GRAY) for im in rgb],fB)
        # NIR camera geometry is tested against measured depths; no unverified
        # assumption that RGB and NIR are perfectly coaligned is used for fusion.
        dn=stereo(*nir,fB)
        x=np.load(BASE/'frames'/f'{i:06d}.npz')['camera'];x=x[np.linalg.norm(x,axis=1)>3.]@r1.T
        sparse,lidar,_,_=dense_depth(x,K,(720,960));mask=sparse>0
        row=dict(frame=i,seconds=time.time()-t,lidar_coverage=float(np.mean(lidar>0)))
        for name,d in [('rgb',dg),('nir',dn)]:
            ok=mask&(d>0);e=np.abs(d[ok]-sparse[ok]);rel=e/sparse[ok]
            row[name]=dict(coverage=float(np.mean(d>0)),lidar_comparisons=int(ok.sum()),
                median_relative_error=float(np.median(rel)),p90_relative_error=float(np.percentile(rel,90)),
                fraction_within_10pct=float(np.mean(rel<.10)),median_error_cm=float(np.median(e)*100))
        # Reject local gross disagreement with LiDAR; RGB is the geometric
        # extension. NIR remains a separate candidate until its results checked.
        np.savez_compressed(dest,rgb=cv2.cvtColor(rgb[0],cv2.COLOR_BGR2RGB),rgb_depth=dg,nir_depth=dn,lidar=lidar,sparse=sparse)
        dump(str(dest.with_suffix('.json').relative_to(OUT)),row);rows.append(row)
        print('stereo_depth',json.dumps(row),flush=True)
        if i in [5,21,26,37,45]:
            color=cv2.applyColorMap(np.uint8(np.clip(dg/25,0,1)*255),cv2.COLORMAP_TURBO);color[dg==0]=0
            cv2.imwrite(str(OUT/f'stereo_depth_{i:02d}.jpg'),np.hstack((rgb[0],color)))
    dump('stereo_depth_metrics.json',rows)

def multiview():
    q=np.load(OUT/'rectified_poses.npz');P,K=q['poses'],q['K'];h,w=720,960
    zz=[dict(np.load(OUT/'stereo'/f'{i:06d}.npz')) for i in range(48)]
    rows=[];(OUT/'hybrid').mkdir(exist_ok=True)
    # Conservative: NIR can add only where it agrees with RGB, and is
    # never used as independent dense geometry with unknown RGB/NIR alignment.
    for i in range(48):
        d=zz[i]['rgb_depth'];yy,xx=np.nonzero(d>0);z=d[yy,xx]
        x=np.column_stack(((xx-K[0,2])/K[0,0]*z,(yy-K[1,2])/K[1,1]*z,z));world=transform(x,P[i])
        support=np.zeros(len(x),np.uint8);contradict=np.zeros(len(x),np.uint8)
        neighbors=[j for j in range(max(0,i-2),min(48,i+3)) if j!=i and j not in HOLD]
        for j in neighbors:
            xc=transform(world,np.linalg.inv(P[j]));uv=xc[:,:2]/np.maximum(xc[:,2:],.01)*[K[0,0],K[1,1]]+K[:2,2]
            ij=np.rint(uv).astype(int);ok=(xc[:,2]>1)&(ij[:,0]>=0)&(ij[:,0]<w)&(ij[:,1]>=0)&(ij[:,1]<h)
            ids=np.flatnonzero(ok);u,v=ij[ids].T;other=zz[j]['rgb_depth'][v,u];lz=zz[j]['lidar'][v,u]
            other=np.where(lz>0,lz,other);has=other>0;delta=xc[ids,2]-other;tol=.10+.025*other
            support[ids]+=has&(np.abs(delta)<tol)
            # A candidate in front of another measured surface violates
            # that view's free space; points behind it may be occluded.
            contradict[ids]+=has&(delta < -np.maximum(.25,.05*other))
        local=zz[i]['lidar'][yy,xx];localok=(local==0)|(np.abs(local-z)<(.08+.025*local))
        good=(support>=2)&(contradict==0)&localok
        verified=np.zeros_like(d);verified[yy[good],xx[good]]=z[good]
        # Measured LiDAR dominates the depth, stereo only fills holes.
        hybrid=zz[i]['lidar'].copy();fill=(hybrid==0)&(verified>0);hybrid[fill]=verified[fill]
        np.savez_compressed(OUT/'hybrid'/f'{i:06d}.npz',rgb=zz[i]['rgb'],depth=hybrid,verified=verified)
        row=dict(frame=i,stereo_pixels=int(len(z)),verified_pixels=int(good.sum()),
            added_pixels=int(fill.sum()),lidar_coverage=float(np.mean(zz[i]['lidar']>0)),hybrid_coverage=float(np.mean(hybrid>0)))
        rows.append(row);print('multiview',json.dumps(row),flush=True)
    dump('multiview_metrics.json',rows)

def correct():
    q=np.load(OUT/'rectified_poses.npz');K=q['K'];s=np.load(OUT/'stereo_calibration.npz');c=dict(np.load(SRC/'calibration.npz'))
    _,_,_,p2,_=rectification(c,s['R'],s['T']);fB=abs(p2[0,3]);rows=[]
    for i in range(48):
        path=OUT/'stereo'/f'{i:06d}.npz';z=dict(np.load(path));row={'frame':i}
        for mod in ['rgb','nir']:
            raw=z.get('raw_'+mod+'_depth',z[mod+'_depth']);z['raw_'+mod+'_depth']=raw
            yy,xx=np.nonzero((raw>0)&(z['sparse']>3)&(z['sparse']<30));measured=z['sparse'][yy,xx];ds=fB/raw[yy,xx];truth=fB/measured
            check=((xx//19+yy//19)%5)==0;train=~check
            def errors(d):
                e=np.abs(fB/np.maximum(d,.1)-measured)/measured
                return float(np.median(e[check])),float(np.mean(e[check]<.1))
            if train.sum()<100 or check.sum()<30:
                row[mod]={'selected':False,'reason':'insufficient anchors'};continue
            fit=least_squares(lambda x:np.r_[(ds[train]*x[0]+x[1]-truth[train]),(x[0]-1)*10],
                [1,0],bounds=([.75,-10],[1.25,10]),loss='soft_l1',f_scale=.5,max_nfev=100)
            old=errors(ds);new=errors(ds*fit.x[0]+fit.x[1]);accept=new[0]<old[0]*.95 and new[1]>=old[1]
            corrected=fB/np.maximum(fB/np.maximum(raw,.001)*fit.x[0]+fit.x[1],.1)
            corrected[(raw==0)|(corrected<1)|(corrected>35)]=0
            z[mod+'_depth']=corrected.astype(np.float32) if accept else raw
            row[mod]=dict(selected=bool(accept),anchors=int(len(ds)),fit_anchors=int(train.sum()),check_anchors=int(check.sum()),
                disparity_scale=float(fit.x[0]),disparity_bias_px=float(fit.x[1]),raw_check_median_relative=old[0],corrected_check_median_relative=new[0],
                raw_check_within_10pct=old[1],corrected_check_within_10pct=new[1])
        np.savez_compressed(path,**z);rows.append(row);print('depth_bias',json.dumps(row),flush=True)
    dump('stereo_bias_correction.json',dict(views=rows,note='Per-view sparse LiDAR fits affine disparity correction with a spatial 80/20 anchor split; six validation frames and their stereo data are excluded from validation-mesh integration and neighbor checks. RGB/NIR raw predictions retained.'))

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['calibrate','depths','correct','multiview']);args=a.parse_args()
    globals()[args.stage]()
