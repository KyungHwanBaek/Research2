#!/usr/bin/env python3
"""Align, measure, and export actual surfaces from photo reconstruction."""
from pathlib import Path
import argparse,json,time
import numpy as np,open3d as o
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from mesh_student_hall_lidar import evaluate
from hall_followup_surface import raw_evaluate,HOLD
from render_student_hall_geometry import render
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/student_hall_geometry_20261031';BASE=ROOT/'experiments/student_hall_lidar'
def dump(path,row):path.write_text(json.dumps(row,indent=2));print(json.dumps(row),flush=True)
def lidar(ids,voxel=.10):
    P=np.load(BASE/'fused/selected_poses.npz')['poses'];parts=[]
    for i in ids:
        x=np.load(BASE/'frames'/f'{i:06d}.npz')['camera'][::2];x=x[np.linalg.norm(x,axis=1)>3];parts.append(x@P[i,:3,:3].T+P[i,:3,3])
    p=o.geometry.PointCloud(o.utility.Vector3dVector(np.concatenate(parts)));return p.voxel_down_sample(voxel)
def assess(mesh):
    z=np.load(BASE/'fused/selected_poses.npz');report=evaluate(mesh,z['poses'],z['K'],HOLD)
    report['hall_interpolated_mae_cm']=float(np.mean([v['clipped_mae_cm'] for v in report['views'] if v['frame']>=20]));report['raw_rays']=raw_evaluate(mesh,z['poses'])
    return report
def mvs(a):
    start=time.time();folder=OUT/f'mvs_{a.variant}_{a.mode}';z=np.load(folder/'alignment.npz');s,r,t=float(z['scale']),z['R'],z['t'];T=np.eye(4);T[:3,:3]=s*r;T[:3,3]=t
    mesh=o.io.read_triangle_mesh(str(folder/'scene_mesh.ply'));pc=o.io.read_point_cloud(str(folder/'scene_dense.ply'));mesh.transform(T);pc.transform(T)
    o.io.write_triangle_mesh(str(folder/'photo_surface_camera_aligned.ply'),mesh)
    train=[i for i in range(48) if i not in HOLD] if a.mode=='validation' else list(range(48));target=lidar(train);source=pc.voxel_down_sample(.10)
    v=np.asarray(source.points);lo=v.min(0)-.5;hi=v.max(0)+.5;x=np.asarray(target.points);target=o.geometry.PointCloud(o.utility.Vector3dVector(x[np.all((x>lo)&(x<hi),axis=1)]));target.estimate_normals(o.geometry.KDTreeSearchParamHybrid(radius=.45,max_nn=30))
    registration=o.pipelines.registration;delta=np.eye(4);history=[]
    for threshold in [.6,.3,.15]:
        result=registration.registration_icp(source,target,threshold,delta,registration.TransformationEstimationPointToPlane(registration.TukeyLoss(k=threshold*.6)),registration.ICPConvergenceCriteria(max_iteration=50))
        delta=result.transformation;history.append(dict(threshold_m=threshold,fitness=float(result.fitness),inlier_rmse_m=float(result.inlier_rmse)))
    angle=float(np.linalg.norm(Rotation.from_matrix(delta[:3,:3]).as_rotvec())*180/np.pi);translation=float(np.linalg.norm(delta[:3,3]));accept=angle<4 and translation<1.5
    if accept:mesh.transform(delta);pc.transform(delta)
    else:delta=np.eye(4)
    mesh.remove_degenerate_triangles();mesh.remove_unreferenced_vertices();mesh.compute_vertex_normals();pc.estimate_normals(o.geometry.KDTreeSearchParamHybrid(radius=.16,max_nn=25))
    o.io.write_triangle_mesh(str(folder/'photo_surface.ply'),mesh);o.io.write_point_cloud(str(folder/'photo_dense_metric.ply'),pc);np.savez(folder/'world_transform.npz',camera_alignment=T,lidar_icp=delta,total=delta@T)
    report=dict(mode=a.mode,variant=a.variant,vertices=len(mesh.vertices),triangles=len(mesh.triangles),dense_photo_points=len(pc.points),alignment_icp=history,icp_translation_m=translation,icp_rotation_deg=angle,icp_accepted=accept,seconds=time.time()-start,geometry_source='Photographs only; a global similarity and rigid ICP to training LiDAR place this geometry into the metric comparison frame. No LiDAR vertices added to photo surface.',caveat='SfM poses reuse all photos. Validation MVS excludes both eyes of withheld scans; ICP target excludes their LiDAR. This is sensor consistency, not surveyed absolute accuracy.')
    if a.mode=='validation':report['validation']=assess(mesh)
    preview=mesh.simplify_quadric_decimation(180000);preview.compute_vertex_normals();o.io.write_triangle_mesh(str(folder/'photo_surface_preview.ply'),preview)
    dump(folder/'surface_report.json',report)
def comparison(a):
    folder=OUT/f'mvs_{a.variant}_{a.mode}';z=np.load(BASE/'fused/selected_poses.npz');ids=[26,34,40]
    sources=[('Existing LiDAR surface',BASE/'fused/student_hall_poisson.ply'),('Photograph-derived surface',folder/'photo_surface.ply')]
    sheet=Image.new('RGB',(1440,780),'white')
    for row,(title,path) in enumerate(sources):
        mesh=o.io.read_triangle_mesh(str(path));mesh.paint_uniform_color([.64,.67,.70]);mesh.compute_triangle_normals()
        for col,i in enumerate(ids):
            image=Image.fromarray(render(mesh,z['poses'][i],z['K'])).resize((480,360));sheet.paste(image,(col*480,row*390+30));d=ImageDraw.Draw(sheet);d.text((col*480+10,row*390+10),f'{title} | camera {i}',fill='black')
    sheet.save(folder/'geometry_comparison.jpg',quality=95)
def neural(a):
    import cv2
    folder=OUT/f'depth_{a.model}_{a.width}';cal=np.load(folder/'calibration.npz');P,K=cal['poses'],cal['K'];h,w=int(cal['height']),int(cal['width'])
    ids=[i for i in range(20,46) if (folder/f'{i:06d}.npz').exists() and (a.mode=='final' or i not in HOLD)];data={i:dict(np.load(folder/f'{i:06d}.npz')) for i in ids};verified={};rows=[]
    for i in ids:
        d=data[i]['depth' if a.corrected else 'raw_depth'];yy,xx=np.indices(d.shape);valid=(d>0);y,x=yy[valid],xx[valid];dep=d[valid]
        xyz=np.column_stack(((x-K[0,2])/K[0,0]*dep,(y-K[1,2])/K[1,1]*dep,dep));world=xyz@P[i,:3,:3].T+P[i,:3,3]
        support=np.zeros(len(x),np.int16);conflict=np.zeros(len(x),np.int16)
        # Distinct timestamps only; same-time right view already checked by LR.
        neighbors=[j for j in ids if j!=i and np.linalg.norm(P[j,:3,3]-P[i,:3,3])<a.radius]
        neighbors=sorted(neighbors,key=lambda j:np.linalg.norm(P[j,:3,3]-P[i,:3,3]))[:8]
        for j in neighbors:
            q=(world-P[j,:3,3])@P[j,:3,:3];uv=q[:,:2]/np.maximum(q[:,2:],.01)*[K[0,0],K[1,1]]+K[:2,2];u,v=uv.T
            dj=data[j]['depth' if a.corrected else 'raw_depth'];ui=np.rint(u).astype(int);vi=np.rint(v).astype(int)
            inside=(q[:,2]>1)&(ui>=0)&(ui<w)&(vi>=0)&(vi<h)
            sample=np.zeros(len(u),np.float32);sample[inside]=dj[vi[inside],ui[inside]];inside &= sample>0
            tolerance=.04+a.relative*q[:,2];support+=inside&(np.abs(sample-q[:,2])<tolerance);conflict+=inside&(q[:,2]<sample-tolerance*2)
        good=(support>=a.support)&(conflict==0);out=np.zeros_like(d);out[y[good],x[good]]=dep[good];verified[i]=out
        rows.append(dict(frame=i,raw_valid=int(valid.sum()),verified=int(good.sum()),fraction=float(good.mean()),neighbors=neighbors));print('MULTIVIEW',rows[-1],flush=True)
    name=f'neural_{a.model}_{a.width}'+('_corrected' if a.corrected else '_raw')+f'_{a.mode}_v{a.voxel:.2f}_s{a.support}_r{a.relative:.3f}';dest=OUT/name;dest.mkdir(exist_ok=True)
    intrinsic=o.camera.PinholeCameraIntrinsic(w,h,K[0,0],K[1,1],K[0,2],K[1,2]);vol=o.pipelines.integration.ScalableTSDFVolume(a.voxel,a.voxel*3,o.pipelines.integration.TSDFVolumeColorType.RGB8)
    for i,d in verified.items():
        rgbd=o.geometry.RGBDImage.create_from_color_and_depth(o.geometry.Image(data[i]['rgb'].copy()),o.geometry.Image(d),depth_scale=1,depth_trunc=35,convert_rgb_to_intensity=False);vol.integrate(rgbd,intrinsic,np.linalg.inv(P[i]))
    mesh=vol.extract_triangle_mesh();mesh.remove_degenerate_triangles();mesh.remove_unreferenced_vertices();clusters,counts,_=mesh.cluster_connected_triangles();counts=np.array(counts)
    if len(counts):mesh.remove_triangles_by_mask(counts[np.array(clusters)]<50)
    mesh.remove_unreferenced_vertices();mesh.compute_vertex_normals();o.io.write_triangle_mesh(str(dest/'surface.ply'),mesh)
    report=dict(name=name,views=rows,vertices=len(mesh.vertices),triangles=len(mesh.triangles),voxel_m=a.voxel,minimum_other_timestamp_support=a.support,relative_depth_tolerance=a.relative,neighbor_radius_m=a.radius,geometry_source='Learned photograph depth with cross-time consistency filtering; LiDAR-derived camera poses. '+('Depth additionally corrected by same-frame LiDAR anchors.' if a.corrected else 'No depth correction by LiDAR anchors.'))
    if a.mode=='validation' and len(mesh.triangles):report['validation']=assess(mesh)
    dump(dest/'surface_report.json',report)
def main(a):globals()[a.stage](a)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['mvs','comparison','neural']);p.add_argument('--mode',choices=['validation','final'],default='validation');p.add_argument('--variant',default='principal');p.add_argument('--model',default='rgb');p.add_argument('--width',type=int,default=960);p.add_argument('--voxel',type=float,default=.06);p.add_argument('--corrected',action='store_true');p.add_argument('--support',type=int,default=2);p.add_argument('--relative',type=float,default=.008);p.add_argument('--radius',type=float,default=7.);main(p.parse_args())
