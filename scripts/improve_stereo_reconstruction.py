#!/usr/bin/env python3
"""LiDAR-free calibrated stereo depth, measured camera tracking and dense fusion.

Only rgb/nir image files and camera calibration are read. Original left.png
images are already undistorted, so rectification uses zero distortion.
Calibration translation is converted from millimetres to metres explicitly.
"""
from __future__ import annotations
import argparse
import json
import time
import hashlib
from pathlib import Path
import cv2
import numpy as np


def write_ply(path, xyz, rgb):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.empty(len(xyz), dtype=[('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1')])
    for k, name in enumerate(('x','y','z')): data[name] = xyz[:,k]
    for k, name in enumerate(('red','green','blue')): data[name] = rgb[:,k]
    with path.open('wb') as f:
        f.write(('ply\nformat binary_little_endian 1.0\nelement vertex %d\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n' % len(data)).encode())
        data.tofile(f)


class Stereo:
    def __init__(self, root, output, num_disp=128):
        self.root, self.output = Path(root), Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.frames = sorted(p for p in self.root.iterdir() if (p/'rgb'/'left.png').exists())
        c = np.load(self.root/'calibration.npz')
        self.K = c['mtx_left'].copy()
        self.size = cv2.imread(str(self.frames[0]/'rgb/left.png')).shape[1::-1]
        self.baseline = float(np.linalg.norm(c['T']) / 1000)
        calibration_path=self.output.parent/'refined_stereo_calibration.npz'
        stereo_R,stereo_T=c['R'],c['T']/1000
        if calibration_path.exists():
            refined=np.load(calibration_path); stereo_R,stereo_T=refined['R'],refined['T']
        self.cache_tag=hashlib.sha256(stereo_R.tobytes()+stereo_T.tobytes()).hexdigest()[:10]
        self.R1,self.R2,self.P1,self.P2,self.Q,_,_ = cv2.stereoRectify(
            self.K, np.zeros(5), c['mtx_right'], np.zeros(5), self.size,
            stereo_R, stereo_T.reshape(3,1), flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
        self.rectK = self.P1[:3,:3]
        self.map1 = cv2.initUndistortRectifyMap(self.K,np.zeros(5),self.R1,self.P1,self.size,cv2.CV_32FC1)
        self.map2 = cv2.initUndistortRectifyMap(c['mtx_right'],np.zeros(5),self.R2,self.P2,self.size,cv2.CV_32FC1)
        params = dict(numDisparities=num_disp,blockSize=5,P1=8*25,P2=32*25,
                      disp12MaxDiff=1,preFilterCap=31,uniquenessRatio=10,
                      speckleWindowSize=100,speckleRange=2,mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        self.left_match = cv2.StereoSGBM_create(minDisparity=0,**params)
        self.right_match = cv2.StereoSGBM_create(minDisparity=-num_disp,**params)
        self.sift = cv2.SIFT_create(nfeatures=3500,contrastThreshold=.015)
        self.bf = cv2.BFMatcher(cv2.NORM_L2)

    def refine_calibration(self):
        from scipy.optimize import least_squares
        c=np.load(self.root/'calibration.npz'); pairs=[]
        for i in [60,180,420,600,900,1080,1350,1500,1740]:
            a=cv2.imread(str(self.frames[i]/'rgb/left.png'),0)
            b=cv2.imread(str(self.frames[i]/'rgb/right.png'),0)
            ka,da=self.sift.detectAndCompute(a,None); kb,db=self.sift.detectAndCompute(b,None)
            good=[m for m,n in self.bf.knnMatch(da,db,k=2) if m.distance<.65*n.distance]
            ua=np.float32([ka[m.queryIdx].pt for m in good]); ub=np.float32([kb[m.trainIdx].pt for m in good])
            F,mask=cv2.findFundamentalMat(ua,ub,cv2.USAC_MAGSAC,.6,.999,10000)
            ua,ub=ua[mask.ravel()>0],ub[mask.ravel()>0]
            pairs.append((ua,ub)); print('calibration_matches',i,len(ua),flush=True)
        ua=np.concatenate([p[0] for p in pairs]); ub=np.concatenate([p[1] for p in pairs])
        xa=cv2.undistortPoints(ua[:,None],self.K,None)[:,0]
        xb=cv2.undistortPoints(ub[:,None],c['mtx_right'],None)[:,0]
        E,mask=cv2.findEssentialMat(xa,xb,np.eye(3),method=cv2.RANSAC,prob=.999,threshold=.0007)
        _,R,t,mask_pose=cv2.recoverPose(E,xa,xb,np.eye(3),mask=mask)
        keep=mask_pose.ravel()>0
        x=np.column_stack((xa[keep],np.ones(keep.sum())))
        y=np.column_stack((xb[keep],np.ones(keep.sum())))
        def residual(v):
            r=cv2.Rodrigues(v[:3])[0]; tt=v[3:]/np.linalg.norm(v[3:])
            tx=np.array([[0,-tt[2],tt[1]],[tt[2],0,-tt[0]],[-tt[1],tt[0],0]])
            e=tx@r; ex=x@e.T; ety=y@e
            return (np.sum(y*ex,axis=1)/np.sqrt(ex[:,0]**2+ex[:,1]**2+ety[:,0]**2+ety[:,1]**2))*self.K[0,0]
        initial=np.r_[cv2.Rodrigues(R)[0].ravel(),t.ravel()]
        fit=least_squares(residual,initial,loss='soft_l1',f_scale=.3,max_nfev=300)
        R=cv2.Rodrigues(fit.x[:3])[0]; T=fit.x[3:]/np.linalg.norm(fit.x[3:])*self.baseline
        target=self.output.parent/'refined_stereo_calibration.npz'
        np.savez(target,R=R,T=T,mtx_left=self.K,mtx_right=c['mtx_right'])
        report={'training_frames':[60,180,420,600,900,1080,1350,1500,1740],
                'independent_test_frames':[30,300,750,1200,1600],
                'matches':len(x),'sampson_median_px':float(np.median(np.abs(residual(fit.x)))),
                'rotation_change_deg':float(np.linalg.norm(cv2.Rodrigues(R@c['R'].T)[0])*180/np.pi),
                'baseline_m':self.baseline,'R':R.tolist(),'T_m':T.tolist(),
                'note':'Baseline magnitude retained; relative rotation/translation direction refined from RGB stereo only. No LiDAR accessed.'}
        target.with_suffix('.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report),flush=True)

    def image(self, i, modality='rgb', right=False):
        side = 'right' if right else 'left'
        im=cv2.imread(str(self.frames[i]/modality/(side+'.png')),cv2.IMREAD_COLOR)
        return cv2.remap(im,*(self.map2 if right else self.map1),cv2.INTER_LINEAR)

    def compute(self, i, modality='rgb'):
        cache=self.output/f'{i:06d}_{modality}_{self.cache_tag}.npz'
        if cache.exists():
            z=np.load(cache); return z['depth'],z['disp'],z['valid']
        l=cv2.cvtColor(self.image(i,modality),cv2.COLOR_BGR2GRAY)
        r=cv2.cvtColor(self.image(i,modality,True),cv2.COLOR_BGR2GRAY)
        dl=self.left_match.compute(l,r).astype(np.float32)/16
        dr=self.right_match.compute(r,l).astype(np.float32)/16
        yy,xx=np.indices(dl.shape,dtype=np.float32)
        xr=xx-dl
        back=cv2.remap(dr,xr,yy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=999)
        valid=(dl>2)&(xr>=0)&(xr<dl.shape[1]-1)&(np.abs(dl+back)<.8)
        # Reject unreliable textureless patches and depth discontinuities.
        mean=cv2.boxFilter(l.astype(np.float32),-1,(5,5))
        var=cv2.boxFilter(l.astype(np.float32)**2,-1,(5,5))-mean**2
        valid &= var>6
        med=cv2.medianBlur(dl,5)
        valid &= np.abs(dl-med)<1
        depth=np.zeros_like(dl)
        depth[valid]=self.rectK[0,0]*self.baseline/dl[valid]
        valid &= (depth>.7)&(depth<35)
        depth[~valid]=0
        np.savez_compressed(cache,depth=depth,disp=dl,valid=valid)
        return depth,dl,valid

    def points(self, depth, uv):
        uv=np.asarray(uv)
        z=depth[np.rint(uv[:,1]).astype(int),np.rint(uv[:,0]).astype(int)]
        xy=(uv-self.rectK[:2,2])/np.array([self.rectK[0,0],self.rectK[1,1]])
        return np.column_stack((xy*z[:,None],z))

    def combined_depth(self,i):
        rgb=self.compute(i,'rgb')[0]; nir=self.compute(i,'nir')[0]
        both=(rgb>0)&(nir>0)
        agree=both&(np.abs(rgb-nir)<(.12*np.minimum(rgb,nir)+.08))
        out=rgb.copy()
        out[agree]=(rgb[agree]+nir[agree])/2
        out[(rgb==0)&(nir>0)]=nir[(rgb==0)&(nir>0)]
        # Independent spectra disagreeing strongly are not trusted.
        out[both&~agree]=0
        return out

    def features(self,i):
        cache=self.output/f'{i:06d}_features_{self.cache_tag}.npz'
        if cache.exists():
            z=np.load(cache); return [(z['uv'+str(j)],z['desc'+str(j)]) for j in range(2)]
        result=[]; data={}
        for j,m in enumerate(['rgb','nir']):
            g=cv2.cvtColor(self.image(i,m),cv2.COLOR_BGR2GRAY)
            kp,desc=self.sift.detectAndCompute(g,None)
            uv=np.array([k.pt for k in kp],np.float32)
            result.append((uv,desc)); data['uv'+str(j)]=uv;data['desc'+str(j)]=desc
        np.savez_compressed(cache,**data)
        return result

    def estimate_motion(self,a,b,fa=None,fb=None):
        fa=self.features(a) if fa is None else fa
        fb=self.features(b) if fb is None else fb
        da=self.combined_depth(a); db=self.combined_depth(b)
        aa=[];bb=[]
        for (ua,desca),(ub,descb) in zip(fa,fb):
            matches=self.bf.knnMatch(desca,descb,k=2)
            good=[m for m,n in matches if m.distance<.72*n.distance]
            if good:
                aa.append(ua[[m.queryIdx for m in good]]);bb.append(ub[[m.trainIdx for m in good]])
        if not aa: return None,{'a':a,'b':b,'reason':'no_matches'}
        ua=np.concatenate(aa);ub=np.concatenate(bb)
        # Do not duplicate RGB/NIR correspondences at the same image location.
        _,ids=np.unique(np.round(ua/2).astype(int),axis=0,return_index=True)
        ua,ub=ua[ids],ub[ids]
        xa=self.points(da,ua); valid=xa[:,2]>0
        obj,img=xa[valid],ub[valid]
        if len(obj)<15: return None,{'a':a,'b':b,'reason':'insufficient_depth','matches':len(obj)}
        ok,r,t,ins=cv2.solvePnPRansac(obj,img,self.rectK,None,iterationsCount=1000,reprojectionError=1.5,confidence=.999,flags=cv2.SOLVEPNP_EPNP)
        if not ok or len(ins)<15: return None,{'a':a,'b':b,'reason':'pnp_failed'}
        ids=ins.ravel(); r,t=cv2.solvePnPRefineLM(obj[ids],img[ids],self.rectK,None,r,t)
        rot=cv2.Rodrigues(r)[0]; T=np.eye(4);T[:3,:3]=rot;T[:3,3]=t.ravel()
        predicted=obj[ids]@rot.T+t.ravel()
        observed=self.points(db,img[ids]); has=observed[:,2]>0
        rel=np.abs(predicted[has,2]-observed[has,2])/np.maximum(observed[has,2],.1)
        # Stereo consistency is checked independently at the target frame.
        consistent=np.ones(len(ids),bool);consistent[has]=rel<.15
        if consistent.sum()>=15:
            ids=ids[consistent];r,t=cv2.solvePnPRefineLM(obj[ids],img[ids],self.rectK,None,r,t)
            T[:3,:3]=cv2.Rodrigues(r)[0];T[:3,3]=t.ravel()
        reproj=cv2.projectPoints(obj[ids],r,t,self.rectK,None)[0][:,0]
        error=np.linalg.norm(reproj-img[ids],axis=1)
        report={'a':a,'b':b,'matches_3d':len(obj),'inliers':len(ids),'median_reprojection_px':float(np.median(error)),
                'target_depth_relative_median':float(np.median(rel)) if len(rel) else None,
                'translation_m':float(np.linalg.norm(t)), 'rotation_deg':float(np.linalg.norm(r)*180/np.pi)}
        if np.linalg.norm(t)>max(1.,.5*abs(b-a)) or np.median(error)>1.5:
            report['reason']='motion_outlier';return None,report
        return T,report

    def track(self,stride=2,limit=0):
        indices=list(range(0,min(limit or len(self.frames),len(self.frames)),stride))
        if indices[-1]!=min(limit or len(self.frames),len(self.frames))-1:
            indices.append(min(limit or len(self.frames),len(self.frames))-1)
        trackpath=self.output.parent/f'tracking_s{stride}_{len(indices)}.npz'
        poses=[np.eye(4)]; edges=[]; reports=[]; components=[0]
        last=indices[0]; prev_features=self.features(last)
        start=time.time()
        for k,i in enumerate(indices[1:],1):
            features=self.features(i)
            T,report=self.estimate_motion(last,i,prev_features,features)
            if T is None and i-last>1:
                # Recover a difficult skipped interval through every actual frame.
                T=np.eye(4);subreports=[]
                for j in range(last,i):
                    sub,sr=self.estimate_motion(j,j+1);subreports.append(sr)
                    if sub is None: T=None;break
                    T=sub@T
                report['fallback']=subreports
            if T is None:
                components.append(components[-1]+1);poses.append(poses[-1].copy());edges.append(np.eye(4))
                report['tracking_break']=True
            else:
                components.append(components[-1]);poses.append(poses[-1]@np.linalg.inv(T));edges.append(T)
            reports.append(report)
            last=i;prev_features=features
            if k%20==0 or k==len(indices)-1:
                np.savez(trackpath,indices=indices[:k+1],poses=poses,edges=edges,components=components)
                trackpath.with_suffix('.json').write_text(json.dumps({'elapsed_s':time.time()-start,'stride':stride,'reports':reports},indent=2))
                print('tracking',k,'/',len(indices)-1,'frame',i,'segments',components[-1]+1,'inliers',report.get('inliers'),'seconds',round(time.time()-start),flush=True)
        print('tracking_saved',trackpath,flush=True)

    def find_loop_bridges(self,trackfile):
        z=np.load(trackfile);indices=z['indices'];segments=z['components']
        candidates=list(range(0,len(indices),3))
        sample=np.concatenate([self.features(int(indices[k]))[0][1][::80] for k in candidates]).astype(np.float32)
        root=lambda x:np.sqrt(x/(x.sum(axis=1,keepdims=True)+1e-8)).astype(np.float32)
        sample=root(sample)
        cv2.setRNGSeed(42)
        _,_,centers=cv2.kmeans(sample,96,None,(cv2.TERM_CRITERIA_MAX_ITER|cv2.TERM_CRITERIA_EPS,30,.001),1,cv2.KMEANS_PP_CENTERS)
        hist=[]
        for k in candidates:
            desc=root(self.features(int(indices[k]))[0][1])
            matches=self.bf.match(desc,centers)
            h=np.bincount([m.trainIdx for m in matches],minlength=len(centers)).astype(float)
            hist.append(h)
        hist=np.array(hist);idf=np.log((len(hist)+1)/(np.sum(hist>0,axis=0)+1))+1
        hist=np.sqrt(hist/(hist.sum(axis=1,keepdims=True)+1e-8))*idf
        hist/=np.linalg.norm(hist,axis=1,keepdims=True)
        reports=[];accepted=[]
        for seg in np.unique(segments)[1:]:
            targets=[k for k in candidates if segments[k]==seg]
            print('loop_search_segment',seg,'target_views',len(targets),flush=True)
            for k in targets:
                row=candidates.index(k)
                earlier=[j for j,kk in enumerate(candidates) if segments[kk]<seg and abs(indices[kk]-indices[k])>30]
                ranked=sorted(earlier,key=lambda j:float(hist[row]@hist[j]),reverse=True)[:6]
                for j in ranked:
                    a,b=int(indices[candidates[j]]),int(indices[k])
                    T,report=self.estimate_motion(a,b)
                    reports.append(report)
                    if T is not None and report['inliers']>=40 and report['inliers']/report['matches_3d']>.25 and report['median_reprojection_px']<.8 and (report['target_depth_relative_median'] or 1)<.07:
                        reverse,rr=self.estimate_motion(b,a)
                        if reverse is not None:
                            cycle=reverse@T
                            angle=np.linalg.norm(cv2.Rodrigues(cycle[:3,:3])[0])*180/np.pi
                            if np.linalg.norm(cycle[:3,3])<.3 and angle<2:
                                accepted.append({'a':a,'b':b,'T':T.tolist(),'inliers':report['inliers'],'cycle_translation_m':float(np.linalg.norm(cycle[:3,3])),'cycle_rotation_deg':float(angle)})
                                print('verified_loop_bridge',accepted[-1],flush=True)
                                break
                if any(x['b']==int(indices[k]) for x in accepted):break
        path=self.output.parent/'loop_bridges.json'
        path.write_text(json.dumps({'accepted':accepted,'attempts':reports},indent=2))
        print('loop_search_done','verified',len(accepted),'attempts',len(reports),flush=True)

    def diagnostics(self, indices):
        from PIL import Image,ImageDraw
        rows=[]; stats=[]
        for i in indices:
            panels=[]; info={'frame':i,'timestamp':self.frames[i].name}
            l=self.image(i); r=self.image(i,right=True)
            kl,dl=self.sift.detectAndCompute(cv2.cvtColor(l,cv2.COLOR_BGR2GRAY),None)
            kr,dr=self.sift.detectAndCompute(cv2.cvtColor(r,cv2.COLOR_BGR2GRAY),None)
            good=[a for a,b in self.bf.knnMatch(dl,dr,k=2) if a.distance<.65*b.distance]
            dy=np.array([abs(kl[m.queryIdx].pt[1]-kr[m.trainIdx].pt[1]) for m in good])
            info['stereo_matches']=len(good)
            info['epipolar_vertical_median_px']=float(np.median(dy))
            info['epipolar_vertical_p90_px']=float(np.percentile(dy,90))
            for m in ['rgb','nir']:
                depth,disp,valid=self.compute(i,m)
                color=cv2.applyColorMap(np.uint8(np.clip(disp,0,70)/70*255),cv2.COLORMAP_TURBO)
                color[~valid]=0
                info[m+'_valid_fraction']=float(valid.mean())
                info[m+'_median_depth_m']=float(np.median(depth[valid]))
                panels.append(np.concatenate((self.image(i,m),color),axis=1))
            rgbdepth=self.compute(i)[0]; ndepth=self.compute(i,'nir')[0]
            overlap=(rgbdepth>0)&(ndepth>0)
            info['rgb_nir_depth_relative_median']=float(np.median(np.abs(rgbdepth[overlap]-ndepth[overlap])/rgbdepth[overlap]))
            print(json.dumps(info),flush=True); stats.append(info)
            arr=np.concatenate(panels,axis=0)
            im=Image.fromarray(cv2.cvtColor(arr,cv2.COLOR_BGR2RGB)); im.thumbnail((1000,750))
            ImageDraw.Draw(im).text((12,12),f'Frame {i}: RGB / NIR; rectified input | confident disparity',fill='white')
            im.save(self.output/f'diagnostic_{i:06d}.jpg'); rows.append(im)
        (self.output/'diagnostics.json').write_text(json.dumps({'baseline_m':self.baseline,'translation_units_assumed':'mm','rectified_K':self.rectK.tolist(),'frames':stats},indent=2))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',default='data/extracted/sampled_540p/09-05-17-34-36')
    p.add_argument('--output',default='experiments/stereo_improved_0905')
    p.add_argument('--diagnostic',default='30,300,750,1200,1600')
    p.add_argument('--refine-calibration',action='store_true')
    p.add_argument('--track',action='store_true')
    p.add_argument('--loop-search',action='store_true')
    p.add_argument('--stride',type=int,default=2)
    p.add_argument('--limit',type=int,default=0)
    a=p.parse_args(); cv2.setNumThreads(4)
    s=Stereo(a.root,Path(a.output)/'depth')
    if a.refine_calibration:
        s.refine_calibration(); return
    if a.track:
        s.track(a.stride,a.limit);return
    if a.loop_search:
        s.find_loop_bridges(Path(a.output)/'tracking_s3_609.npz');return
    s.diagnostics([int(x) for x in a.diagnostic.split(',')])


if __name__=='__main__': main()
