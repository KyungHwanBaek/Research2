#!/usr/bin/env python3
"""Refine stereo visual odometry with local SfM constraints; fuse tested depths.

Does not silently bridge failed tracking intervals or pretend SfM components
were a single bundle-adjusted model. Pose-graph constraints are recorded.
"""
from pathlib import Path
import argparse
import json
import time
import shutil
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
from improve_stereo_reconstruction import Stereo,write_ply
from export_colmap_viewer import qvec_to_rotmat


def read_model(root):
    points={}
    with (root/'points3D.txt').open() as f:
        for line in f:
            if not line.strip() or line.startswith('#'):continue
            p=line.split();points[int(p[0])]=np.array([float(x) for x in p[1:4]])
    images={}
    with (root/'images.txt').open() as f:
        while line:=f.readline():
            if not line.strip() or line.startswith('#'):continue
            p=line.split();T=np.eye(4)
            T[:3,:3]=qvec_to_rotmat(np.array(p[1:5],float));T[:3,3]=np.array(p[5:8],float)
            observations=np.array(f.readline().split(),float).reshape(-1,3)
            images[int(Path(p[9]).stem)]={'w2c':T,'obs':observations}
    camera=next(x for x in (root/'cameras.txt').read_text().splitlines() if x and not x.startswith('#')).split()
    return images,points,int(camera[2])


def component_constraints(stereo,indices):
    parent=Path('experiments/rgb_colmap_0905_all_s1_360_seq20_fixed/sparse')
    lookup={int(i):k for k,i in enumerate(indices)}
    D=np.eye(4);D[:3,:3]=stereo.R1
    edges=[]; reports=[]; models=[]
    for path in sorted(parent.glob('*_text_for_summary')):
        images,points,width=read_model(path)
        ratios=[];test=[]
        sample=sorted(images)[::15]
        for j,i in enumerate(sample):
            im=images[i];obs=im['obs'];obs=obs[obs[:,2]>=0]
            if len(obs)==0:continue
            pts=np.array([points[int(p)] for p in obs[:,2]])
            camera_xyz=pts@im['w2c'][:3,:3].T+im['w2c'][:3,3]
            rect_xyz=camera_xyz@stereo.R1.T
            uv=obs[:,:2]*(stereo.size[0]/width)
            rect=cv2.undistortPoints(uv[:,None],stereo.K,None,R=stereo.R1,P=stereo.P1)[:,0]
            xy=np.rint(rect).astype(int)
            valid=(xy[:,0]>=0)&(xy[:,0]<stereo.size[0])&(xy[:,1]>=0)&(xy[:,1]<stereo.size[1])&(rect_xyz[:,2]>.01)
            xy=xy[valid];zz=rect_xyz[valid,2]
            depth=stereo.combined_depth(i)[xy[:,1],xy[:,0]]
            valid=depth>0
            vals=depth[valid]/zz[valid]
            (ratios if j%3 else test).extend(vals.tolist())
        if len(ratios)<20:
            print('skip_scale_insufficient',path.name,len(ratios),flush=True);continue
        ratios=np.array(ratios); scale=float(np.median(ratios))
        robust=np.abs(np.log(ratios/scale))<.2
        scale=float(np.median(ratios[robust]))
        held=np.array(test)
        report={'component':path.name,'registered_frames':len(images),'scale_m_per_sfm_unit':scale,
                'training_depth_samples':len(ratios),'training_inlier_fraction':float(robust.mean()),
                'heldout_depth_samples':len(held),'heldout_depth_relative_median':float(np.median(np.abs(scale/held-1))) if len(held) else None}
        report['accepted_constraint']=bool(robust.mean()>=.3 and (report['heldout_depth_relative_median'] or 0)<.35)
        reports.append(report);print('component_scale',json.dumps(report),flush=True)
        if not report['accepted_constraint']:continue
        model={}
        for i,im in images.items():
            T=im['w2c'].copy();T[:3,3]*=scale
            model[i]=np.linalg.inv(D@T)
        models.append((path.name,model))
        valid_indices=[i for i in indices if i in model]
        for offset in [1,5,15]:
            for a,b in zip(valid_indices,valid_indices[offset:]):
                if b-a>offset*5:continue
                relative=np.linalg.inv(model[b])@model[a]
                edges.append((lookup[int(a)],lookup[int(b)],relative,'sfm'))
    return edges,reports,models


def ground_measurements(stereo,indices):
    cache=stereo.output.parent/'ground_planes.npz'
    if cache.exists():
        z=np.load(cache);return z['normals'],z['heights'],z['confidence']
    rng=np.random.default_rng(1337);normals=[];heights=[];confidence=[]
    for j,i in enumerate(indices):
        d=stereo.combined_depth(int(i));yy,xx=np.mgrid[300:540:4,70:650:4]
        uv=np.column_stack((xx.ravel(),yy.ravel())).astype(np.float32)
        pts=stereo.points(d,uv);pts=pts[(pts[:,2]>1)&(pts[:,2]<15)]
        best=np.zeros(len(pts),bool)
        for _ in range(100):
            if len(pts)<100:break
            p=pts[rng.choice(len(pts),3,replace=False)]
            n=np.cross(p[1]-p[0],p[2]-p[0]);n/=max(np.linalg.norm(n),1e-12)
            if n[1]<0:n=-n
            h=n@p[0]
            if n[1]<.85 or not .4<h<2.:continue
            ins=np.abs(pts@n-h)<.04
            if ins.sum()>best.sum():best=ins
        if best.sum()>=200 and best.mean()>.35:
            center=pts[best].mean(axis=0);_,_,vt=np.linalg.svd(pts[best]-center,full_matrices=False)
            n=vt[-1]
            if n[1]<0:n=-n
            normals.append(n);heights.append(float(n@center));confidence.append(float(best.mean()))
        else:normals.append([0,1,0]);heights.append(0.);confidence.append(0.)
        if j%100==0:print('ground_estimation',j,'/',len(indices),flush=True)
    np.savez(cache,normals=normals,heights=heights,confidence=confidence)
    return np.array(normals),np.array(heights),np.array(confidence)


def optimize(stereo,trackfile,ground_prior=False):
    z=np.load(trackfile);indices=z['indices'];poses=z['poses'];segments=z['components']
    sfm,reports,models=component_constraints(stereo,indices)
    repaired=[];track_edges=z['edges'].copy()
    for seg in np.unique(segments):
        mask=segments==seg;first=np.flatnonzero(mask)[0]
        poses[mask]=np.linalg.inv(poses[first])@poses[mask]
    loopfile=stereo.output.parent/'loop_bridges.json'
    loops=json.loads(loopfile.read_text())['accepted'] if loopfile.exists() else []
    lookup={int(i):k for k,i in enumerate(indices)}
    loop_edges=[]
    for loop in loops:
        a,b=lookup[loop['a']],lookup[loop['b']];T=np.array(loop['T'])
        sa,sb=segments[a],segments[b]
        if sa!=sb:
            change=poses[a]@np.linalg.inv(T)@np.linalg.inv(poses[b])
            mask=segments==sb;poses[mask]=change@poses[mask];segments[mask]=sa
        loop_edges.append((a,b,T,'stereo'))
        repaired.append(loop)
    edges=[]
    for k,T in enumerate(track_edges):
        if z['components'][k]==z['components'][k+1]:edges.append((k,k+1,T,'stereo'))
    edges+=loop_edges
    # Constraints from a sparse model cannot by themselves certify a gap bridge.
    sfm=[e for e in sfm if segments[e[0]]==segments[e[1]]]
    edges+=sfm
    for k in range(0,len(indices)-15,10):
        for off in [5,15]:
            if segments[k]!=segments[k+off]:continue
            T,info=stereo.estimate_motion(int(indices[k]),int(indices[k+off]))
            if T is not None and info['inliers']>=35 and info['median_reprojection_px']<.9 and (info['target_depth_relative_median'] or 1)<.1:
                edges.append((k,k+off,T,'stereo'))
    if ground_prior and (stereo.output.parent/'optimized_poses.npz').exists():
        current=stereo.output.parent/'optimized_poses.npz'
        for suffix in ['.npz','.json']:
            backup=current.with_name('optimized_poses_unconstrained'+suffix)
            if not backup.exists():shutil.copy2(current.with_suffix(suffix),backup)
        previous=np.load(current)
        if np.array_equal(previous['components'],segments):poses=previous['poses']
    # Each actual tracking segment gets its own fixed gauge. No fake connection.
    anchors=set(np.flatnonzero(np.r_[True,np.diff(segments)!=0]).tolist())
    variable=[i for i in range(len(poses)) if i not in anchors]
    vlookup={i:k for k,i in enumerate(variable)}
    x0=np.column_stack((Rotation.from_matrix(poses[variable,:3,:3]).as_rotvec(),poses[variable,:3,3])).ravel()
    aa=np.array([e[0] for e in edges]);bb=np.array([e[1] for e in edges])
    meas=np.array([e[2] for e in edges]);is_sfm=np.array([e[3]=='sfm' for e in edges])
    # SfM constrains rotation and longer-range shape; stereo supplies metric translation.
    r_sigma=np.where(is_sfm,.004,.012)
    t_sigma=np.where(is_sfm,.1+.3*np.linalg.norm(meas[:,:3,3],axis=1),.045)
    floor_ids=np.array([],int);floor_targets=np.empty((0,3));floor_levels=np.array([]);floor_normals=np.empty((0,3));floor_heights=np.array([])
    if ground_prior:
        normals,heights,confidence=ground_measurements(stereo,indices)
        floor_ids=np.flatnonzero(confidence>.35);floor_normals=normals[floor_ids];floor_heights=heights[floor_ids]
        target={};level={}
        for seg in np.unique(segments):
            ids=np.flatnonzero((segments==seg)&(confidence>.35));anchor=ids[0]
            target[seg]=poses[anchor,:3,:3]@normals[anchor]
            level[seg]=target[seg]@poses[anchor,:3,3]+heights[anchor]
        floor_targets=np.array([target[segments[k]] for k in floor_ids]);floor_levels=np.array([level[segments[k]] for k in floor_ids])
    def unpack(x):
        result=poses.copy();v=x.reshape(-1,6)
        result[variable,:3,:3]=Rotation.from_rotvec(v[:,:3]).as_matrix();result[variable,:3,3]=v[:,3:]
        return result
    def residual(x):
        p=unpack(x);ra=p[aa,:3,:3];rb=p[bb,:3,:3]
        pred_r=np.einsum('nji,njk->nik',rb,ra)
        pred_t=np.einsum('nji,nj->ni',rb,p[aa,:3,3]-p[bb,:3,3])
        err_r=Rotation.from_matrix(np.einsum('nji,njk->nik',meas[:,:3,:3],pred_r)).as_rotvec()/r_sigma[:,None]
        err_t=(pred_t-meas[:,:3,3])/t_sigma[:,None]
        base=np.column_stack((err_r,err_t)).ravel()
        if len(floor_ids):
            world_normals=np.einsum('nij,nj->ni',p[floor_ids,:3,:3],floor_normals)
            gravity=np.cross(world_normals,floor_targets)/.018
            height=(np.sum(p[floor_ids,:3,3]*floor_targets,axis=1)+floor_heights-floor_levels)/.18
            base=np.r_[base,np.column_stack((gravity,height)).ravel()]
        return base
    sparsity=lil_matrix((len(edges)*6+len(floor_ids)*4,len(x0)),dtype=int)
    for k,(a,b,_,_) in enumerate(edges):
        for i in [a,b]:
            if i in vlookup:sparsity[k*6:(k+1)*6,vlookup[i]*6:(vlookup[i]+1)*6]=1
    for j,i in enumerate(floor_ids):
        if i in vlookup:sparsity[len(edges)*6+j*4:len(edges)*6+(j+1)*4,vlookup[i]*6:(vlookup[i]+1)*6]=1
    print('pose_graph_start','nodes',len(poses),'edges',len(edges),'anchors',len(anchors),flush=True)
    before=residual(x0)
    fit=least_squares(residual,x0,jac_sparsity=sparsity.tocsr(),loss='soft_l1',f_scale=2.,max_nfev=200,ftol=1e-4,verbose=1)
    refined=unpack(fit.x)
    out=stereo.output.parent/'optimized_poses.npz'
    np.savez(out,indices=indices,poses=refined,components=segments)
    report={'nodes':len(poses),'stereo_edges':len(edges)-len(sfm),'sfm_edges':len(sfm),'tracking_segments':len(anchors),
            'initial_robust_cost':float(np.sum(2*(np.sqrt(1+(before/2)**2)-1))),
            'final_robust_cost':float(np.sum(2*(np.sqrt(1+(residual(fit.x)/2)**2)-1))),
            'solver_success':bool(fit.success),'solver_message':fit.message,'evaluations':fit.nfev,'measured_gap_bridges':repaired,
            'components':reports,'ground_prior':bool(ground_prior),'ground_plane_constraints':len(floor_ids),
            'ground_height_median_m':float(np.median(floor_heights)) if len(floor_ids) else None,
            'note':'Image-only pose graph. No LiDAR or ground-truth trajectory. Metric scale assumes calibration T in mm; drift is not independently measured. Ground-prior variant additionally assumes approximately level traversable ground within each tracking segment.'}
    out.with_suffix('.json').write_text(json.dumps(report,indent=2))
    np.savez(stereo.output.parent/'sfm_constraint_edges.npz',a=aa,b=bb,relative=meas,is_sfm=is_sfm)
    print('optimized',out,flush=True)


def voxel_average(xyz,rgb,voxel=.045):
    keys=np.floor(xyz/voxel).astype(np.int32)
    _,inv,counts=np.unique(keys,axis=0,return_inverse=True,return_counts=True)
    pts=np.column_stack([np.bincount(inv,weights=xyz[:,j])/counts for j in range(3)])
    col=np.column_stack([np.bincount(inv,weights=rgb[:,j])/counts for j in range(3)])
    return pts.astype(np.float32),col.astype(np.uint8),counts


def export_viewer(path,xyz,rgb,cameras,title,limit=140000):
    from export_colmap_viewer import HTML_TEMPLATE
    rng=np.random.default_rng(31);ids=rng.choice(len(xyz),min(limit,len(xyz)),replace=False)
    pts=np.column_stack((xyz[ids],rgb[ids])).round(4).tolist()
    payload={'points':pts,'cameras':np.asarray(cameras).round(4).tolist()}
    html=HTML_TEMPLATE.replace('{{','{').replace('}}','}')
    html=html.replace('__TITLE__',title).replace('__POINT_COUNT__',f'{len(ids):,} / {len(xyz):,}').replace('__DATA_JSON__',json.dumps(payload,separators=(',',':')))
    html=html.replace('estimated COLMAP camera trajectory','stereo + SfM camera trajectory').replace('value="1.4"','value="1.0"')
    Path(path).write_text(html)


def fuse(stereo,posefile,stride=2,limit=0,output=None):
    z=np.load(posefile);indices=z['indices'];poses=z['poses'];segments=z['components']
    out=Path(output) if output else stereo.output.parent/'fused';out.mkdir(parents=True,exist_ok=True)
    allstats=[]
    for segment in np.unique(segments):
        ids=np.flatnonzero(segments==segment)[::stride]
        if limit:ids=ids[:limit]
        chunks=[];colors=[];stats=[];total=0
        for step,k in enumerate(ids):
            i=int(indices[k]);depth=stereo.combined_depth(i)
            yy,xx=np.mgrid[0:depth.shape[0]:2,0:depth.shape[1]:2]
            uv=np.column_stack((xx.ravel(),yy.ravel())).astype(np.float32)
            xyz=stereo.points(depth,uv);valid=(xyz[:,2]>0)&(xyz[:,2]<28)
            xyz,uv=xyz[valid],uv[valid]
            world=xyz@poses[k,:3,:3].T+poses[k,:3,3]
            support=np.zeros(len(world),np.uint8)
            tested=np.zeros(len(world),np.uint8)
            relative_errors=[]
            for delta in [-2,2]:
                n=k+delta
                if n<0 or n>=len(indices) or segments[n]!=segment:continue
                dn=stereo.combined_depth(int(indices[n]))
                cam=(world-poses[n,:3,3])@poses[n,:3,:3]
                proj=cam[:,:2]/np.maximum(cam[:,2:],1e-5)*np.array([stereo.rectK[0,0],stereo.rectK[1,1]])+stereo.rectK[:2,2]
                xy=np.rint(proj).astype(int)
                inside=(cam[:,2]>.5)&(xy[:,0]>=0)&(xy[:,0]<dn.shape[1])&(xy[:,1]>=0)&(xy[:,1]<dn.shape[0])
                ix=np.flatnonzero(inside);observed=dn[xy[ix,1],xy[ix,0]]
                has=observed>0;ix=ix[has];observed=observed[has]
                err=np.abs(cam[ix,2]-observed)/observed
                tested[ix]+=1;support[ix]+=(err<.055).astype(np.uint8)
                relative_errors.extend(err[err<.2].tolist())
            keep=support>=1
            full_color=np.ascontiguousarray(stereo.image(i)[:,:,::-1])
            color=full_color[uv[:,1].astype(int),uv[:,0].astype(int)]
            chunks.append(world[keep].astype(np.float32));colors.append(color[keep]);total+=keep.sum()
            stats.append({'frame':i,'stereo_points':len(world),'temporal_consistent_points':int(keep.sum()),
                          'cross_view_depth_relative_median':float(np.median(relative_errors)) if relative_errors else None})
            if step%25==0:print('fusing',int(segment),step,'/',len(ids),'points',int(total),flush=True)
        if not total:continue
        xyz=np.concatenate(chunks);rgb=np.concatenate(colors)
        xyz,rgb,counts=voxel_average(xyz,rgb)
        # Statistical outlier removal, with distances estimated from the local cloud.
        tree=cKDTree(xyz);dist,_=tree.query(xyz,k=9,workers=4)
        means=dist[:,1:].mean(axis=1);threshold=float(np.median(means)+3*np.std(means))
        keep=means<threshold;xyz,rgb=xyz[keep],rgb[keep]
        stem=out/f'stereo_rgb_nir_segment{int(segment)}'
        write_ply(stem.with_suffix('.ply'),xyz,rgb)
        np.savez_compressed(stem.with_suffix('.npz'),xyz=xyz,rgb=rgb)
        export_viewer(stem.with_suffix('.html'),xyz,rgb,poses[ids,:3,3],f'RGB + NIR stereo reconstruction · segment {segment}')
        report={'segment':int(segment),'frames_fused':len(ids),'source_frame_range':[int(indices[ids[0]]),int(indices[ids[-1]])],
                'points_before_voxel':int(total),'final_points':len(xyz),'voxel_size_m':.045,'max_depth_m':28,
                'temporal_relative_tolerance':.055,'frames':stats}
        if stem.with_suffix('.json').exists():
            old=json.loads(stem.with_suffix('.json').read_text())
            report.update({k:v for k,v in old.items() if k.startswith('mesh_')})
        stem.with_suffix('.json').write_text(json.dumps(report,indent=2));allstats.append(report)
        print('fused_saved',stem,'points',len(xyz),flush=True)
    (out/'summary.json').write_text(json.dumps(allstats,indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='data/extracted/sampled_540p/09-05-17-34-36')
    p.add_argument('--output',default='experiments/stereo_improved_0905')
    p.add_argument('--trackfile',default='experiments/stereo_improved_0905/tracking_s3_609.npz')
    p.add_argument('--optimize',action='store_true');p.add_argument('--fuse',action='store_true')
    p.add_argument('--ground-prior',action='store_true')
    p.add_argument('--poses',type=Path)
    p.add_argument('--fusion-directory',type=Path)
    p.add_argument('--stride',type=int,default=2);p.add_argument('--limit',type=int,default=0)
    a=p.parse_args();cv2.setNumThreads(4);s=Stereo(a.root,Path(a.output)/'depth')
    if a.optimize:optimize(s,Path(a.trackfile),a.ground_prior)
    if a.fuse:fuse(s,a.poses or s.output.parent/'optimized_poses.npz',a.stride,a.limit,a.fusion_directory)


if __name__=='__main__':main()
