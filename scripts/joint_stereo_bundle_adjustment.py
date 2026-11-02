#!/usr/bin/env python3
"""Native stereo-rig bundle adjustment with measured RGB image features.

Uses actual left/right SIFT coordinates, calibrated metric baseline, temporal
feature tracks and robust Ceres BA. Dense stereo depth is used for initialization
and separate dense consistency evaluation, not as a fabricated right observation.
"""
from pathlib import Path
from collections import defaultdict
from functools import lru_cache
import argparse,json,time,subprocess
import cv2
import numpy as np
import pycolmap as pc
from improve_stereo_reconstruction import Stereo


def right_features(s,i):
    path=s.output/f'{i:06d}_right_features_{s.cache_tag}.npz'
    if path.exists():
        z=np.load(path);return z['uv'],z['desc']
    image=cv2.cvtColor(s.image(i,right=True),cv2.COLOR_BGR2GRAY)
    kp,desc=s.sift.detectAndCompute(image,None)
    uv=np.array([p.pt for p in kp],np.float32)
    np.savez_compressed(path,uv=uv,desc=desc);return uv,desc


def build_model(s,indices,poses,folder,offsets=(1,2,5)):
    folder.mkdir(parents=True,exist_ok=True)
    feature=[];right=[];stereo_links=[];offsets_nodes=[0]
    for k,i in enumerate(indices):
        uv,desc=s.features(int(i))[0];ur,dr=right_features(s,int(i))
        good=[m for m,n in s.bf.knnMatch(desc,dr,k=2) if m.distance<.7*n.distance]
        xyz=s.points(s.compute(int(i),'rgb')[0],uv)
        links={}
        for m in good:
            a,b=uv[m.queryIdx],ur[m.trainIdx];d=a[0]-b[0]
            if abs(a[1]-b[1])>1.2 or not 2.5<d<130:continue
            z=s.rectK[0,0]*s.baseline/d
            if xyz[m.queryIdx,2]>0 and abs(z-xyz[m.queryIdx,2])<.08*z:
                links[m.queryIdx]=m.trainIdx
        feature.append((uv,desc,xyz));right.append((ur,dr));stereo_links.append(links)
        offsets_nodes.append(offsets_nodes[-1]+len(uv))
        if k%50==0:print('stereo_features',k,'/',len(indices),'stereo_pairs',len(links),flush=True)
    parent=np.arange(offsets_nodes[-1],dtype=np.int32)
    def find(a):
        while parent[a]!=a:parent[a]=parent[parent[a]];a=int(parent[a])
        return a
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b:parent[max(a,b)]=min(a,b)
    matched=set();edge_stats=[]
    for k in range(len(indices)):
        ua,da,xa=feature[k]
        for off in offsets:
            j=k+off
            if j>=len(indices):continue
            ub,db,xb=feature[j]
            good=[m for m,n in s.bf.knnMatch(da,db,k=2) if m.distance<.7*n.distance]
            if len(good)<20:continue
            pa=np.array([ua[m.queryIdx] for m in good]);pb=np.array([ub[m.trainIdx] for m in good])
            F,mask=cv2.findFundamentalMat(pa,pb,cv2.USAC_MAGSAC,1.,.999,3000)
            if mask is None:continue
            count=0
            # Reprojection and independent target depth reject repeated structure.
            T=np.linalg.inv(poses[j])@poses[k]
            for m,inside in zip(good,mask.ravel()):
                if not inside:continue
                x=xa[m.queryIdx];y=xb[m.trainIdx]
                if x[2]<=0 or y[2]<=0:continue
                pred=T[:3,:3]@x+T[:3,3]
                if pred[2]<=0:continue
                pixel=pred[:2]/pred[2]*[s.rectK[0,0],s.rectK[1,1]]+s.rectK[:2,2]
                if np.linalg.norm(pixel-ub[m.trainIdx])>5+off or abs(pred[2]-y[2])>.15*y[2]:continue
                a=offsets_nodes[k]+m.queryIdx;b=offsets_nodes[j]+m.trainIdx
                union(a,b);matched.add(a);matched.add(b);count+=1
            edge_stats.append([int(indices[k]),int(indices[j]),count])
        if k%50==0:print('temporal_tracks',k,'/',len(indices),flush=True)
    groups=defaultdict(list)
    for node in matched:groups[find(node)].append(node)
    rec=pc.Reconstruction()
    params=np.array([s.rectK[0,0],s.rectK[1,1],s.rectK[0,2],s.rectK[1,2]])
    for cid in [1,2]:rec.add_camera(pc.Camera(camera_id=cid,model='PINHOLE',width=s.size[0],height=s.size[1],params=params))
    rig=pc.Rig(rig_id=1);ref=pc.sensor_t(pc.SensorType.CAMERA,1);other=pc.sensor_t(pc.SensorType.CAMERA,2)
    rig.add_ref_sensor(ref);rig.add_sensor(other,pc.Rigid3d(pc.Rotation3d(),np.array([-s.baseline,0.,0.])))
    rec.add_rig(rig)
    for k,i in enumerate(indices):
        frame=pc.Frame(frame_id=k+1,rig_id=1,rig_from_world=pc.Rigid3d(np.linalg.inv(poses[k])[:3]))
        frame.add_data_id(pc.data_t(ref,2*k+1));frame.add_data_id(pc.data_t(other,2*k+2))
        rec.add_frame(frame)
        for side,cid,coords in [('left',1,feature[k][0]),('right',2,right[k][0])]:
            im=pc.Image(name=f'{side}/{int(i):06d}.png',keypoints=coords.astype(float),camera_id=cid,image_id=2*k+cid)
            im.frame_id=k+1;rec.add_image(im)
        rec.register_frame(k+1)
    created=0;rejected=0
    @lru_cache(maxsize=12)
    def color_image(i):return s.image(i)
    for nodes in groups.values():
        if len(nodes)<2:continue
        ks=np.searchsorted(offsets_nodes,nodes,side='right')-1
        if len(np.unique(ks))!=len(ks):continue
        local=np.array(nodes)-np.array(offsets_nodes)[ks]
        worlds=[]
        for k,l in zip(ks,local):
            x=feature[k][2][l]
            if x[2]>0:worlds.append(poses[k,:3,:3]@x+poses[k,:3,3])
        if not worlds:continue
        world=np.median(worlds,axis=0);track=pc.Track();has_stereo=0;left_count=0
        for k,l in zip(ks,local):
            cam=(world-poses[k,:3,3])@poses[k,:3,:3]
            if cam[2]<=.5:continue
            uv=cam[:2]/cam[2]*[s.rectK[0,0],s.rectK[1,1]]+s.rectK[:2,2]
            if np.linalg.norm(uv-feature[k][0][l])>10:continue
            track.add_element(2*int(k)+1,int(l));left_count+=1
            if int(l) in stereo_links[k]:
                r=stereo_links[k][int(l)];uv[0]-=s.rectK[0,0]*s.baseline/cam[2]
                if np.linalg.norm(uv-right[k][0][r])<10:
                    track.add_element(2*int(k)+2,int(r));has_stereo+=1
        if left_count<2 or has_stereo<1:rejected+=1;continue
        k,l=int(ks[0]),int(local[0]);uv=feature[k][0][l].astype(int)
        color=color_image(int(indices[k]))[uv[1],uv[0],::-1].copy()
        # A descriptor may match the same right feature twice; don't reuse it.
        elems=list(track.elements)
        if any(rec.images[e.image_id].points2D[e.point2D_idx].has_point3D() for e in elems):continue
        rec.add_point3D(world,track,color);created+=1
    print('model_ready','frames',len(indices),'points',created,'rejected',rejected,flush=True)
    (folder/'initial').mkdir(exist_ok=True);rec.write(str(folder/'initial'))
    (folder/'track_edges.json').write_text(json.dumps(edge_stats))
    return rec


def dense_score(s,indices,poses):
    fractions=[];medians=[]
    for k in range(0,len(indices)-2,5):
        j=k+2;depth=s.combined_depth(int(indices[k]));other=s.combined_depth(int(indices[j]))
        yy,xx=np.mgrid[30:510:8,30:690:8];uv=np.column_stack((xx.ravel(),yy.ravel())).astype(np.float32)
        xyz=s.points(depth,uv);xyz=xyz[xyz[:,2]>0]
        T=np.linalg.inv(poses[j])@poses[k];cam=xyz@T[:3,:3].T+T[:3,3]
        pix=cam[:,:2]/np.maximum(cam[:,2:],1e-5)*[s.rectK[0,0],s.rectK[1,1]]+s.rectK[:2,2]
        xy=np.rint(pix).astype(int);valid=(cam[:,2]>.5)&(xy[:,0]>=0)&(xy[:,0]<720)&(xy[:,1]>=0)&(xy[:,1]<540)
        xx2=xy[valid];d=other[xx2[:,1],xx2[:,0]];zz=cam[valid,2];ok=d>0
        err=np.abs(d[ok]-zz[ok])/d[ok]
        if len(err):fractions.append(float(np.mean(err<.055)));medians.append(float(np.median(err)))
    return {'median_cross_view_relative_depth_error':float(np.median(medians)),'mean_cross_view_agreement_fraction':float(np.mean(fractions)),'evaluated_pairs':len(fractions)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--limit',type=int,default=0);p.add_argument('--segment',type=int,default=-1)
    p.add_argument('--iterations',type=int,default=70);p.add_argument('--local-render',action='store_true');a=p.parse_args();cv2.setNumThreads(4)
    s=Stereo('data/extracted/sampled_540p/09-05-17-34-36','experiments/stereo_improved_0905/depth')
    old=np.load('experiments/stereo_improved_0905/optimized_poses.npz');allposes=old['poses'].copy();reports=[]
    if a.local_render:
        source=Path('exports/3dgs_0905_comparison_sparse')
        original=pc.Reconstruction(str(source/'sparse/0'))
        manifest=json.loads((source/'manifest.json').read_text());indices=np.array(manifest['images'])
        lookup={int(Path(im.name).stem):im for im in original.images.values()}
        poses=[]
        for i in indices:
            w2c=np.eye(4);w2c[:3]=lookup[int(i)].cam_from_world().matrix();poses.append(np.linalg.inv(w2c))
        old={'indices':indices,'poses':np.array(poses),'components':np.zeros(len(indices),int)}
        allposes=old['poses'].copy()
    out=Path('experiments/stereo_joint_ba_0905');out.mkdir(exist_ok=True)
    for seg in np.unique(old['components']):
        if a.segment>=0 and seg!=a.segment:continue
        selected=np.flatnonzero(old['components']==seg)
        if a.limit:selected=selected[:a.limit]
        indices=old['indices'][selected];poses=old['poses'][selected];folder=out/f'segment{seg}'
        if a.limit:folder=out/f'pilot_segment{seg}_{a.limit}'
        if a.local_render:folder=out/'local_render'
        rec=build_model(s,indices,poses,folder)
        rec.update_point_3d_errors()
        initial_error=rec.compute_mean_reprojection_error();before=dense_score(s,indices,poses)
        opt=pc.BundleAdjustmentOptions();opt.refine_focal_length=False;opt.refine_principal_point=False;opt.refine_extra_params=False;opt.refine_sensor_from_rig=False
        opt.ceres.loss_function_type=pc.LossFunctionType.SOFT_L1;opt.ceres.loss_function_scale=1.
        opt.ceres.solver_options.max_num_iterations=a.iterations;opt.ceres.solver_options.num_threads=6
        opt.ceres.solver_options.function_tolerance=1e-5
        cfg=pc.BundleAdjustmentConfig()
        for image_id in rec.images:cfg.add_image(image_id)
        cfg.set_constant_rig_from_world_pose(1)
        solver=pc.create_default_bundle_adjuster(opt,cfg,rec);start=time.time();summary=solver.solve()
        refined=np.array([np.linalg.inv(rec.frames[k+1].rig_from_world.matrix().tolist()+[[0,0,0,1]]) for k in range(len(indices))])
        rec.update_point_3d_errors()
        after=dense_score(s,indices,refined)
        (folder/'refined').mkdir(exist_ok=True);rec.write(str(folder/'refined'))
        report={'segment':int(seg),'frames':len(indices),'points':rec.num_points3D(),'initial_reprojection_px':initial_error,
                'final_reprojection_px':rec.compute_mean_reprojection_error(),'dense_before':before,'dense_after':after,'elapsed_s':time.time()-start,
                'summary':summary.todict() if hasattr(summary,'todict') else str(summary)}
        (folder/'metrics.json').write_text(json.dumps(report,indent=2,default=str));reports.append(report)
        np.savez(folder/'poses.npz',indices=indices,poses=refined,components=np.full(len(indices),seg))
        if a.local_render:
            from scipy.spatial.transform import Rotation
            export=Path('exports/3dgs_0905_comparison_joint');text=export/'sparse_text';text.mkdir(parents=True,exist_ok=True)
            (export/'sparse/0').mkdir(parents=True,exist_ok=True)
            shared=Path('exports/3dgs_stereo_0905_images').resolve()
            if not (export/'images').exists():(export/'images').symlink_to(shared,target_is_directory=True)
            (text/'cameras.txt').write_text(f'1 PINHOLE {s.size[0]} {s.size[1]} '+ ' '.join(str(float(x)) for x in [s.rectK[0,0],s.rectK[1,1],s.rectK[0,2],s.rectK[1,2]])+'\n')
            lines=[]
            for k,(i,P) in enumerate(zip(indices,refined)):
                T=np.linalg.inv(P);q=Rotation.from_matrix(T[:3,:3]).as_quat();q=np.r_[q[3],q[:3]]
                lines.append(f'{k+1} '+' '.join(str(float(x)) for x in np.r_[q,T[:3,3]])+f' 1 {int(i):06d}.png\n\n')
            (text/'images.txt').write_text(''.join(lines))
            (text/'points3D.txt').write_text(''.join(f'{k+1} '+' '.join(str(float(x)) for x in point.xyz)+' '+' '.join(str(int(x)) for x in point.color)+' 0\n' for k,point in enumerate(rec.points3D.values())))
            subprocess.run(['colmap','model_converter','--input_path',str(text),'--output_path',str(export/'sparse/0'),'--output_type','BIN'],check=True)
            (export/'manifest.json').write_text(json.dumps({'images':indices.tolist(),'initial_points':rec.num_points3D(),'method':'joint_stereo_BA','evaluation':'same 61 left images, every 8th excluded from 3DGS training; all images used for camera geometry as in prior SfM baseline'},indent=2))
        # Keep improved poses only if separate dense pixels confirm the change.
        if after['median_cross_view_relative_depth_error']<before['median_cross_view_relative_depth_error'] and after['mean_cross_view_agreement_fraction']>=before['mean_cross_view_agreement_fraction']:
            allposes[selected]=refined;report['selected']=True
        else:report['selected']=False
        print('BA_RESULT',json.dumps(report,default=str),flush=True)
    if a.local_render:return
    if not a.limit:
        np.savez(out/'selected_poses.npz',indices=old['indices'],poses=allposes,components=old['components'])
        (out/'comparison.json').write_text(json.dumps(reports,indent=2,default=str))


if __name__=='__main__':main()
