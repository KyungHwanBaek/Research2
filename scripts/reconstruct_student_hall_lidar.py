#!/usr/bin/env python3
"""Measured RGB/COLMAP + raw LiDAR experiments for the 48-view student hall.

This is a project-specific fusion pipeline, not a reproduction of Colmap-PCD.
Coordinates: camera x right/y down/z forward; world anchored to camera zero.
LiDAR coordinates are metres; cam2lidar translation is converted mm -> m.
"""
from pathlib import Path
import argparse
import json
import time
import cv2
import h5py
import numpy as np
from scipy.spatial import cKDTree, Delaunay
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from functools import lru_cache

ROOT = Path('data/extracted/01-26-15-41-24')
OUT = Path('experiments/student_hall_lidar')

def dump(path, obj):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False))

def transform(x,T): return x@T[:3,:3].T+T[:3,3]

def cloud(x, voxel=0.):
    import open3d as o
    p=o.geometry.PointCloud(o.utility.Vector3dVector(np.asarray(x,dtype=np.float64)))
    if voxel:p=p.voxel_down_sample(voxel)
    return p

def dense_depth(x,K,shape):
    """Rasterize nearest returns, then interpolate only small smooth triangles."""
    h,w=shape
    x=x[(x[:,2]>.6)&(x[:,2]<45)]
    uv=x[:,:2]/x[:,2:]*[K[0,0],K[1,1]]+K[:2,2]
    ok=(uv[:,0]>1)&(uv[:,0]<w-2)&(uv[:,1]>1)&(uv[:,1]<h-2)
    x,uv=x[ok],uv[ok]
    pixels=np.rint(uv).astype(int); order=np.argsort(x[:,2])
    _,ids=np.unique(pixels[order,1]*w+pixels[order,0],return_index=True)
    ids=order[ids];x,uv,pixels=x[ids],uv[ids],pixels[ids]
    sparse=np.zeros((h,w),np.float32);sparse[pixels[:,1],pixels[:,0]]=x[:,2]
    tri=Delaunay(uv)
    yy,xx=np.indices((h,w));grid=np.column_stack((xx.ravel(),yy.ravel()))
    simplex=tri.find_simplex(grid);inside=simplex>=0; si=simplex[inside]
    vertices=tri.simplices[si];zs=x[vertices,2];us=uv[vertices]
    edges=np.max(np.linalg.norm(us-np.roll(us,1,axis=1),axis=2),axis=1)
    good=(edges<22)&((zs.max(1)-zs.min(1))<(.15+.045*zs.min(1)))
    si=si[good];grid_ids=np.flatnonzero(inside)[good]
    b=np.einsum('ijk,ik->ij',tri.transform[si,:2],grid[grid_ids]-tri.transform[si,2])
    bary=np.column_stack((b,1-b.sum(1)))
    # Reciprocal depth is linear on planar projected triangles.
    depth=np.zeros(h*w,np.float32)
    depth[grid_ids]=1/np.sum(bary/zs[good],axis=1)
    depth=depth.reshape(h,w);depth[sparse>0]=sparse[sparse>0]
    return sparse,depth,uv,x

def prepare():
    OUT.mkdir(parents=True,exist_ok=True)
    frames=sorted(p for p in ROOT.iterdir() if p.is_dir() and (p/'rgb').exists())
    c=np.load(ROOT/'calibration.npz');K=c['mtx_left'].copy();K[:2]*=960/1440
    T=np.load(ROOT/'cam2lidar.npy');T[:3,3]/=1000
    T[3]=[0,0,0,1]
    np.savez(OUT/'calibration.npz',K=K,cam2lidar=T,shape=[720,960])
    (OUT/'images').mkdir(exist_ok=True);(OUT/'frames').mkdir(exist_ok=True)
    sift=cv2.SIFT_create(nfeatures=6500,contrastThreshold=.012)
    maps=cv2.initUndistortRectifyMap(c['mtx_left'],c['dist_left'],None,K,(960,720),cv2.CV_32FC1)
    previews=[]
    with h5py.File(ROOT/'storage.hdf5') as f:
        for i,frame in enumerate(frames):
            target=OUT/'frames'/f'{i:06d}.npz'
            if target.exists():continue
            im=cv2.imread(str(frame/'rgb/left_distorted.png'))
            im=cv2.remap(im,*maps,cv2.INTER_LINEAR)
            cv2.imwrite(str(OUT/'images'/f'{i:06d}.png'),im)
            x=f['frame'][frame.name]['lidar/points'][:].reshape(-1,3)
            norm=np.linalg.norm(x,axis=1)
            x=x[np.isfinite(x).all(1)&(norm>1)&(norm<45)]
            xc=transform(x,np.linalg.inv(T))
            sparse,depth,uv,visible=dense_depth(xc,K,im.shape[:2])
            kp,desc=sift.detectAndCompute(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),None)
            kpuv=np.array([k.pt for k in kp],np.float32)
            np.savez_compressed(target,lidar=x.astype(np.float32),camera=xc.astype(np.float32),
                sparse=sparse,depth=depth,uv=kpuv,desc=desc)
            if i in [0,16,26,40,47]:
                overlay=im.copy();z=visible[:,2];color=cv2.applyColorMap(np.uint8(np.clip(z/25,0,1)*255),cv2.COLORMAP_TURBO).reshape(-1,3)
                for q,col in zip(np.rint(uv).astype(int)[::2],color[::2]):cv2.circle(overlay,tuple(q),2,tuple(map(int,col)),-1)
                cv2.putText(overlay,f'Frame {i} | camera<-LiDAR initial calibration',(16,35),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2)
                cv2.imwrite(str(OUT/f'overlay_{i:02d}.jpg'),overlay)
            print('prepare',i,'lidar',len(x),'depth',round(np.mean(depth>0),3),flush=True)
    dump(OUT/'source.json',{'root':str(ROOT),'frames':[p.name for p in frames],
        'friend_map_present':False,'note':'Uses original dataset raw LiDAR, not a friend-provided map.'})

@lru_cache(maxsize=48)
def data(i):
    with np.load(OUT/'frames'/f'{i:06d}.npz') as z:
        return {k:z[k] for k in z.files}

def pixels_xyz(depth, uv, K):
    uv=np.asarray(uv);xy=np.rint(uv).astype(int)
    z=depth[xy[:,1],xy[:,0]]
    return np.column_stack(((uv-K[:2,2])/[K[0,0],K[1,1]]*z[:,None],z))

def umeyama(x,y):
    mx,my=x.mean(0),y.mean(0);xx=x-mx;yy=y-my
    u,s,v=np.linalg.svd(yy.T@xx/len(x));D=np.ones(3);D[-1]=np.linalg.det(u@v)
    r=(u*D)@v;scale=np.sum(s*D)/np.mean(np.sum(xx**2,axis=1))
    return scale,r,my-scale*r@mx

def sfm():
    import pycolmap
    K=np.load(OUT/'calibration.npz')['K'];rows=[];rng=np.random.default_rng(42)
    for component in [0,1]:
        rec=pycolmap.Reconstruction(f'experiments/rgb_colmap_well_lit_outdoor_768_exhaustive_fixed/sparse/{component}')
        ratios=[];observations=[];poses={}
        for im in rec.images.values():
            i=int(Path(im.name).stem);d=data(i)['depth'];T=im.cam_from_world().matrix()
            Tw=np.eye(4);Tw[:3]=T;poses[i]=np.linalg.inv(Tw)
            for p in im.points2D:
                if not p.has_point3D():continue
                uv=p.xy*960/768;x=rec.points3D[p.point3D_id].xyz
                z=(T[:3,:3]@x+T[:3,3])[2];ix,iy=np.rint(uv).astype(int)
                if 0<=ix<960 and 0<=iy<720 and d[iy,ix]>0 and z>0:
                    ratios.append(d[iy,ix]/z);observations.append([i,*uv,*x,d[iy,ix],z])
        ratios=np.array(ratios);scale=float(np.median(ratios))
        rows.append({'component':component,'images':sorted(poses),'depth_observations':len(ratios),'scale':scale,
                     'relative_depth_median':float(np.median(np.abs(ratios/scale-1))),
                     'depth_ratio_percentiles':np.percentile(ratios,[10,50,90]).tolist()})
        ids=np.array(sorted(poses));ps=np.array([poses[i] for i in ids]);ps[:,:3,3]*=scale
        np.savez(OUT/f'colmap_component{component}.npz',indices=ids,poses=ps,scale=scale,observations=observations)
    dump(OUT/'colmap_depth_alignment.json',rows);print(json.dumps(rows,indent=2),flush=True)

def pnp_pair(a,b):
    za,zb=data(a),data(b);K=np.load(OUT/'calibration.npz')['K']
    matches=cv2.BFMatcher().knnMatch(za['desc'],zb['desc'],k=2)
    good=[m for m,n in matches if m.distance<.7*n.distance]
    if len(good)<12:return None,{'matches':len(good)}
    ua=za['uv'][[m.queryIdx for m in good]];ub=zb['uv'][[m.trainIdx for m in good]]
    xa=pixels_xyz(za['depth'],ua,K);valid=xa[:,2]>0
    xa,ub=xa[valid],ub[valid]
    if len(xa)<12:return None,{'matches':len(xa)}
    ok,r,t,ins=cv2.solvePnPRansac(xa,ub,K,None,iterationsCount=2000,reprojectionError=3.,confidence=.999,flags=cv2.SOLVEPNP_EPNP)
    if not ok or ins is None or len(ins)<12:return None,{'matches':len(xa),'failure':'pnp'}
    ids=ins.ravel();r,t=cv2.solvePnPRefineLM(xa[ids],ub[ids],K,None,r,t)
    T=np.eye(4);T[:3,:3]=cv2.Rodrigues(r)[0];T[:3,3]=t.ravel()
    err=np.linalg.norm(cv2.projectPoints(xa[ids],r,t,K,None)[0][:,0]-ub[ids],axis=1)
    return T,{'matches':len(xa),'inliers':len(ids),'pixel_median':float(np.median(err))}

def kiss():
    from kiss_icp.kiss_icp import KissICP
    from kiss_icp.config import KISSConfig
    c=KISSConfig();c.data.deskew=False;c.data.min_range=1.;c.data.max_range=40.
    c.mapping.voxel_size=.25;c.registration.max_num_threads=4
    k=KissICP(c);ps=[];E=np.load(OUT/'calibration.npz')['cam2lidar']
    for i in range(48):
        k.register_frame(data(i)['lidar'].astype(float),np.array([],dtype=float))
        ps.append(np.linalg.inv(E)@k.last_pose@E)
        print('kiss',i,np.round(ps[-1][:3,3],2),flush=True)
    np.savez(OUT/'kiss_poses.npz',indices=np.arange(48),poses=ps)

def icp(a,b,T,min_range=0.):
    import open3d as o
    est=o.pipelines.registration.TransformationEstimationPointToPlane
    for voxel,dist,it in [(.45,1.4,50),(.20,.65,40),(.10,.25,30)]:
        xa=data(a)['camera'][::2];xb=data(b)['camera'][::2]
        xa=xa[np.linalg.norm(xa,axis=1)>min_range];xb=xb[np.linalg.norm(xb,axis=1)>min_range]
        pa=cloud(xa,voxel);pb=cloud(xb,voxel)
        pb.estimate_normals(o.geometry.KDTreeSearchParamHybrid(radius=voxel*3,max_nn=30))
        result=o.pipelines.registration.registration_icp(pa,pb,dist,T,est(o.pipelines.registration.TukeyLoss(dist/2)),
                    o.pipelines.registration.ICPConvergenceCriteria(max_iteration=it))
        T=result.transformation.copy()
    return T,{'fitness':result.fitness,'rmse':result.inlier_rmse}

def global_init(a,b):
    import open3d as o
    reg=o.pipelines.registration;pcs=[];fs=[]
    for i in [a,b]:
        p=cloud(data(i)['camera'][::2],.45)
        p.estimate_normals(o.geometry.KDTreeSearchParamHybrid(radius=1.35,max_nn=30))
        f=reg.compute_fpfh_feature(p,o.geometry.KDTreeSearchParamHybrid(radius=2.25,max_nn=100))
        pcs.append(p);fs.append(f)
    r=reg.registration_ransac_based_on_feature_matching(*pcs,*fs,True,.9,
        reg.TransformationEstimationPointToPoint(False),3,
        [reg.CorrespondenceCheckerBasedOnEdgeLength(.9),reg.CorrespondenceCheckerBasedOnDistance(.9)],
        reg.RANSACConvergenceCriteria(100000,.999))
    return r.transformation

def rescue():
    path=OUT/'registration_edges.json';records=json.loads(path.read_text())
    for r in records:
        if r['fitness']>=.52 or r['b']-r['a']>2:continue
        T=global_init(r['a'],r['b']);T,metric=icp(r['a'],r['b'],T)
        candidate={**metric,'initialization':'FPFH_RANSAC','T':T.tolist()}
        r['candidates'].append(candidate)
        if metric['fitness']>r['fitness']:
            r.update(T=T.tolist(),**metric)
        print('rescue',r['a'],r['b'],candidate['fitness'],'best',r['fitness'],flush=True)
        dump(path,records)

def register():
    import open3d as o
    paths=[np.load(OUT/f'colmap_component{j}.npz') for j in [0,1]]
    cp=[{int(i):P for i,P in zip(z['indices'],z['poses'])} for z in paths]
    kp=np.load(OUT/'kiss_poses.npz')['poses']
    # Establish one metric COLMAP baseline using actual shared camera poses.
    base=cp[0];extra=cp[1];shared=sorted(set(base)&set(extra))
    R=base[0][:3,:3]@extra[0][:3,:3].T
    t=base[0][:3,3]-R@extra[0][:3,3]
    A=np.eye(4);A[:3,:3]=R;A[:3,3]=t
    for i,P in extra.items():
        if i not in base:base[i]=A@P
    anchor=np.linalg.inv(base[0]);base={i:anchor@P for i,P in base.items()}
    ids=sorted(base);np.savez(OUT/'colmap_metric_poses.npz',indices=ids,poses=[base[i] for i in ids])
    cache=OUT/'registration_edges.json'
    records=json.loads(cache.read_text()) if cache.exists() else []
    done={(r['a'],r['b']) for r in records}
    pairs=[(i,i+1) for i in range(47)]+[(i,i+2) for i in range(46)]
    pairs+= [(a,b) for a in range(3) for b in range(20,26)]
    for a,b in pairs:
        if (a,b) in done:continue
        visual,vr=pnp_pair(a,b)
        starts=[('kiss',np.linalg.inv(kp[b])@kp[a])]
        if visual is not None:starts.append(('lidar_depth_pnp',visual))
        if a in base and b in base:starts.append(('colmap',np.linalg.inv(base[b])@base[a]))
        # Refine every plausible independent initialization and retain geometric fit.
        candidates=[]
        for name,T in starts:
            T,r=icp(a,b,T);r.update({'initialization':name,'T':T.tolist()});candidates.append(r)
        candidates.sort(key=lambda r:(-r['fitness'],r['rmse']))
        row={'a':a,'b':b,'candidates':candidates,'visual':vr,'T':candidates[0]['T'],
             'fitness':candidates[0]['fitness'],'rmse':candidates[0]['rmse']}
        if visual is not None:row['visual_T']=visual.tolist()
        records.append(row);dump(cache,records)
        print('pair',a,b,'selected',candidates[0]['initialization'],'fit',round(row['fitness'],3),'rmse',round(row['rmse'],3),'visual',vr,flush=True)
    poses=[np.eye(4)]
    for i in range(47):
        r=next(r for r in records if r['a']==i and r['b']==i+1)
        if r['fitness']<.25:raise RuntimeError(f'Unsafe registration bridge {i}: {r}')
        poses.append(poses[-1]@np.linalg.inv(r['T']))
    np.savez(OUT/'icp_odometry_poses.npz',indices=np.arange(48),poses=poses)
    pg=o.pipelines.registration.PoseGraph()
    for P in poses:pg.nodes.append(o.pipelines.registration.PoseGraphNode(P))
    accepted=[]
    for r in records:
        a,b=r['a'],r['b'];T=np.array(r['T'])
        if r['fitness']<(.25 if b-a==1 else .48) or r['rmse']>.15:continue
        # Reject nonsequential constraints incompatible with visual rotation.
        if 'visual_T' in r:
            V=np.array(r['visual_T']);angle=Rotation.from_matrix(T[:3,:3]@V[:3,:3].T).magnitude()
            if angle>.15 and b-a>1:continue
        pa=cloud(data(a)['camera'][::3],.16);pb=cloud(data(b)['camera'][::3],.16)
        info=o.pipelines.registration.get_information_matrix_from_point_clouds(pa,pb,.25,T)
        pg.edges.append(o.pipelines.registration.PoseGraphEdge(a,b,T,info,uncertain=b-a>1))
        accepted.append([a,b])
    o.pipelines.registration.global_optimization(pg,o.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        o.pipelines.registration.GlobalOptimizationOption(max_correspondence_distance=.25,edge_prune_threshold=.25,reference_node=0))
    np.savez(OUT/'posegraph_poses.npz',indices=np.arange(48),poses=[n.pose for n in pg.nodes])
    dump(OUT/'posegraph.json',{'edges':accepted,'method':'multiscale point-to-plane ICP + LiDAR-depth PnP + COLMAP initialization + robust pose graph'})

def evaluate():
    """Consistency on returns excluded from ICP; this is not survey ground truth."""
    methods=['kiss','colmap_metric','icp_odometry','posegraph','joint','sensor_clean']
    import open3d as o
    pcs=[];normals=[];trees=[];sources=[]
    for i in range(48):
        x=data(i)['camera'];p=cloud(x[::2],.12)
        p.estimate_normals(o.geometry.KDTreeSearchParamHybrid(radius=.5,max_nn=30))
        pcs.append(np.asarray(p.points));normals.append(np.asarray(p.normals));trees.append(cKDTree(pcs[-1]))
        sources.append(np.asarray(cloud(x[1::2],.18).points))
    baseline=np.load(OUT/'colmap_metric_poses.npz');common=set(baseline['indices'].tolist())
    pairs=[(i,i+1) for i in range(47) if i in common and i+1 in common]
    rows=[];details=[]
    for name in methods:
        path=OUT/f'{name}_poses.npz'
        if not path.exists():continue
        z=np.load(path);poses={int(i):P for i,P in zip(z['indices'],z['poses'])}
        dists=[];planes=[]
        for a,b in pairs:
            T=np.linalg.inv(poses[b])@poses[a];xx=transform(sources[a],T)
            dist,nn=trees[b].query(xx,workers=4);plane=np.abs(np.sum((xx-pcs[b][nn])*normals[b][nn],axis=1))
            dists.append(dist);planes.append(plane)
            details.append({'method':name,'a':a,'b':b,'overlap_25cm':float(np.mean(dist<.25)),
                            'clipped_plane_mean_cm':float(np.minimum(plane,.5).mean()*100)})
        d=np.concatenate(dists);pl=np.concatenate(planes)
        row={'method':name,'frames':len(poses),'common_pairs':len(pairs),'heldout_returns':len(d),
            'overlap_25cm':float(np.mean(d<.25)),'point_distance_median_cm':float(np.median(d)*100),
            'clipped_plane_mean_cm':float(np.minimum(pl,.5).mean()*100),
            'plane_median_matched_cm':float(np.median(pl[d<.25])*100)}
        rows.append(row);print(row,flush=True)
    dump(OUT/'evaluation.json',{'metrics':rows,'pairs':details,
         'note':'All methods compared on identical adjacent pairs available in original COLMAP. Odd returns excluded from ICP fitting; target maps use even returns. This is cross-scan consistency, not independent absolute accuracy. FPFH rescue uses downsampled even returns.'})

def sensor_clean():
    import open3d as o
    poses=np.load(OUT/'posegraph_poses.npz')['poses'];accepted=json.loads((OUT/'posegraph.json').read_text())['edges']
    records=json.loads((OUT/'registration_edges.json').read_text());pg=o.pipelines.registration.PoseGraph();rows=[]
    for P in poses:pg.nodes.append(o.pipelines.registration.PoseGraphNode(P))
    for a,b in accepted:
        T=np.linalg.inv(poses[b])@poses[a];T,r=icp(a,b,T,min_range=3.)
        xa=data(a)['camera'][::2];xb=data(b)['camera'][::2]
        pa=cloud(xa[np.linalg.norm(xa,axis=1)>3.],.16);pb=cloud(xb[np.linalg.norm(xb,axis=1)>3.],.16)
        info=o.pipelines.registration.get_information_matrix_from_point_clouds(pa,pb,.25,T)
        pg.edges.append(o.pipelines.registration.PoseGraphEdge(a,b,T,info,uncertain=b-a>1))
        rows.append({'a':a,'b':b,**r});print('clean_registration',a,b,r,flush=True)
    o.pipelines.registration.global_optimization(pg,o.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
       o.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
       o.pipelines.registration.GlobalOptimizationOption(max_correspondence_distance=.25,edge_prune_threshold=.25,reference_node=0))
    np.savez(OUT/'sensor_clean_poses.npz',indices=np.arange(48),poses=[n.pose for n in pg.nodes]);dump(OUT/'sensor_clean.json',rows)

def joint():
    """Joint robust reprojection + LiDAR tangent plane refinement, with pose priors."""
    import pycolmap
    import open3d as o
    from scipy.sparse import lil_matrix
    z=np.load(OUT/'posegraph_poses.npz');poses=z['poses'];K=np.load(OUT/'calibration.npz')['K']
    locals=[];normals=[];trees=[]
    for i in range(48):
        p=cloud(data(i)['camera'][::2],.12)
        p.estimate_normals(o.geometry.KDTreeSearchParamHybrid(radius=.45,max_nn=30))
        locals.append(np.asarray(p.points));normals.append(np.asarray(p.normals));trees.append(cKDTree(locals[-1]))
    X=[];obs=[];constraints=[];colors=[]
    for comp in [0,1]:
        rec=pycolmap.Reconstruction(f'experiments/rgb_colmap_well_lit_outdoor_768_exhaustive_fixed/sparse/{comp}')
        for point in rec.points3D.values():
            track=[];samples=[];planes=[]
            for e in point.track.elements:
                im=rec.images[e.image_id];i=int(Path(im.name).stem);uv=im.points2D[e.point2D_idx].xy*960/768
                track.append((i,uv))
                cam=pixels_xyz(data(i)['depth'],uv[None],K)[0]
                if cam[2]>0:
                    distance,j=trees[i].query(cam)
                    if distance<.25:
                        world=transform(cam[None],poses[i])[0];normal=poses[i,:3,:3]@normals[i][j]
                        samples.append(world);planes.append((world,normal))
            if len(track)<3:continue
            # Triangulate with the currently measured metric camera poses.
            A=[]
            for i,uv in track:
                T=np.linalg.inv(poses[i]);P=K@T[:3]
                A.extend([uv[0]*P[2]-P[0],uv[1]*P[2]-P[1]])
            _,_,v=np.linalg.svd(np.asarray(A));x=v[-1,:3]/v[-1,3]
            if samples:
                med=np.median(samples,axis=0)
                if np.linalg.norm(x-med)>1.5:x=med
            if not np.isfinite(x).all():continue
            checks=[];positive=[]
            for i,uv in track:
                xc=transform(x[None],np.linalg.inv(poses[i]))[0]
                positive.append(xc[2]>.3)
                checks.append(np.linalg.norm(xc[:2]/max(xc[2],.1)*[K[0,0],K[1,1]]+K[:2,2]-uv))
            if np.mean(positive)<.9 or np.median(checks)>15:continue
            n=len(X);X.append(x);colors.append(point.color)
            obs.extend((i,n,*uv) for i,uv in track)
            constraints.extend((n,*q,*normal) for q,normal in planes[::max(1,len(planes)//3)])
    X=np.asarray(X);obs=np.array(obs);planes=np.array(constraints)
    cams=obs[:,0].astype(int);points=obs[:,1].astype(int);uv=obs[:,2:]
    # Reserved observations diagnose refinement, not a fresh-view benchmark.
    val=np.arange(len(obs))%7==0;train=~val
    base=np.array([np.r_[Rotation.from_matrix(np.linalg.inv(P)[:3,:3]).as_rotvec(),np.linalg.inv(P)[:3,3]] for P in poses])
    x0=np.r_[base.ravel(),X.ravel()];npid=planes[:,0].astype(int)
    def residual(v,validation=False):
        pars=v[:48*6].reshape(48,6);xx=v[48*6:].reshape(-1,3);r=Rotation.from_rotvec(pars[:,:3]).as_matrix()
        xc=np.einsum('nij,nj->ni',r[cams],xx[points])+pars[cams,3:]
        pred=xc[:,:2]/np.maximum(xc[:,2:],.1)*[K[0,0],K[1,1]]+K[:2,2]
        if validation:return np.linalg.norm(pred[val]-uv[val],axis=1)
        a=((pred[train]-uv[train])/2).ravel()
        b=np.sum((xx[npid]-planes[:,1:4])*planes[:,4:7],axis=1)/.12
        # Priors constrain changes to lidar registration; use world translation for comparable weights.
        centers=-np.einsum('nji,nj->ni',r,pars[:,3:])
        c=(centers-poses[:,:3,3])/.2
        d=(pars[:,:3]-base[:,:3])/.012
        return np.r_[a,b,c.ravel(),d.ravel()]
    nobs=int(train.sum());nres=nobs*2+len(planes)+48*6
    sparsity=lil_matrix((nres,len(x0)),dtype=int)
    for j,k in enumerate(np.flatnonzero(train)):
        sparsity[j*2:j*2+2,cams[k]*6:cams[k]*6+6]=1
        sparsity[j*2:j*2+2,48*6+points[k]*3:48*6+points[k]*3+3]=1
    off=nobs*2
    for j,pid in enumerate(npid):sparsity[off+j,48*6+pid*3:48*6+pid*3+3]=1
    off+=len(planes)
    for i in range(48):
        sparsity[off+i*3:off+i*3+3,i*6:i*6+6]=1
        sparsity[off+144+i*3:off+144+i*3+3,i*6:i*6+6]=1
    before=residual(x0,True);start=time.time()
    print('joint_start','points',len(X),'observations',len(obs),'planes',len(planes),'validation_median_px',np.median(before),flush=True)
    fit=least_squares(residual,x0,jac_sparsity=sparsity.tocsr(),loss='soft_l1',f_scale=1.,
          x_scale='jac',max_nfev=600,verbose=2,ftol=2e-4)
    pars=fit.x[:288].reshape(48,6);out=[]
    for p in pars:
        T=np.eye(4);T[:3,:3]=Rotation.from_rotvec(p[:3]).as_matrix();T[:3,3]=p[3:];out.append(np.linalg.inv(T))
    np.savez(OUT/'joint_poses.npz',indices=np.arange(48),poses=out,points=fit.x[288:].reshape(-1,3),rgb=colors)
    after=residual(fit.x,True)
    dump(OUT/'joint_report.json',{'points':len(X),'observations':len(obs),'lidar_plane_constraints':len(planes),
        'reserved_observations':int(val.sum()),'before_median_px':float(np.median(before)),'after_median_px':float(np.median(after)),
        'before_p90_px':float(np.percentile(before,90)),'after_p90_px':float(np.percentile(after,90)),
        'nfev':fit.nfev,'success':fit.success,'termination':fit.message,'elapsed_s':time.time()-start,
        'note':'Custom project implementation, not Colmap-PCD reproduction. Camera poses and COLMAP landmarks jointly optimized with fixed LiDAR tangent-plane constraints and LiDAR pose priors. Reserved reprojections are excluded from this refinement, but were used in original COLMAP.'})

def export():
    import open3d as o
    from improve_stereo_reconstruction import write_ply
    evaluation=json.loads((OUT/'evaluation.json').read_text())['metrics']
    candidates=[r for r in evaluation if r['method'] in ['posegraph','joint','sensor_clean']]
    selected=min(candidates,key=lambda r:r['clipped_plane_mean_cm'])['method']
    original=next(r for r in candidates if r['method']=='posegraph')
    best=next(r for r in candidates if r['method']==selected)
    # Keep the established result when another score differs only at numerical precision.
    if original['clipped_plane_mean_cm']-best['clipped_plane_mean_cm']<.01:selected='posegraph'
    ps=np.load(OUT/f'{selected}_poses.npz')['poses'];K=np.load(OUT/'calibration.npz')['K']
    # Full 360-degree LiDAR map, rather than only camera-facing returns.
    xyz=np.concatenate([transform(data(i)['camera'][np.linalg.norm(data(i)['camera'],axis=1)>3.],ps[i]) for i in range(48)])
    p=cloud(xyz,.07);p,ids=p.remove_statistical_outlier(nb_neighbors=20,std_ratio=2.5)
    xyz=np.asarray(p.points);best=np.zeros(len(xyz));rgb=np.full((len(xyz),3),135,np.uint8)
    for i in range(48):
        cam=transform(xyz,np.linalg.inv(ps[i]));uv=cam[:,:2]/np.maximum(cam[:,2:],.001)*[K[0,0],K[1,1]]+K[:2,2]
        pix=np.rint(uv).astype(int);ok=(cam[:,2]>.6)&(cam[:,2]<40)&(pix[:,0]>=0)&(pix[:,0]<960)&(pix[:,1]>=0)&(pix[:,1]<720)
        ids=np.flatnonzero(ok);px=pix[ids];dep=data(i)['depth'][px[:,1],px[:,0]]
        visible=(dep>0)&(np.abs(cam[ids,2]-dep)<(.12+.015*dep))
        ids=ids[visible];px=px[visible]
        score=(1./(1+cam[ids,2]**2/100))*np.maximum(.1,1-np.sum(((uv[ids]-[480,360])/[600,450])**2,axis=1))
        use=score>best[ids];ids,px=ids[use],px[use];best[ids]=score[use]
        im=cv2.imread(str(OUT/'images'/f'{i:06d}.png'))[:,:,::-1];rgb[ids]=im[px[:,1],px[:,0]]
        print('color',i,'covered',int(np.sum(best>0)),flush=True)
    folder=OUT/'fused';folder.mkdir(exist_ok=True)
    write_ply(folder/'lidar_full_map.ply',xyz,rgb)
    colored=best>0;write_ply(folder/'student_hall_colored.ply',xyz[colored],rgb[colored])
    np.savez_compressed(folder/'student_hall_colored.npz',xyz=xyz[colored].astype(np.float32),rgb=rgb[colored])
    np.savez(folder/'selected_poses.npz',indices=np.arange(48),poses=ps,K=K,selected=selected)
    # Compact per-frame RGB-D data is read by the working Open3D 0.19 mesh environment.
    for i in range(48):
        np.savez_compressed(folder/f'rgbd_{i:02d}.npz',depth=data(i)['depth'],
            rgb=cv2.imread(str(OUT/'images'/f'{i:06d}.png'))[:,:,::-1])
    dump(folder/'summary.json',{'selected':selected,'frames':48,'full_map_points':len(xyz),'colored_points':int(colored.sum()),
         'voxel_m':.07,'source':'dataset raw LiDAR + original RGB + original COLMAP components',
         'note':'Unobserved LiDAR geometry remains gray in full map; colored output contains camera-visible geometry only.'})

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','sfm','kiss','register','rescue','evaluate','joint','export','sensor_clean']);a=p.parse_args()
    cv2.setNumThreads(4)
    globals()[a.stage]()

if __name__=='__main__':main()
