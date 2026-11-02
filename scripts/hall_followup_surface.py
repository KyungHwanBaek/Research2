#!/usr/bin/env python3
"""Six virtual LiDAR-centred cameras per scan for 360-degree TSDF fusion.

The virtual faces contain the same raw scan, not six independent observations.
Six entire scans withheld for validation; poses estimated using all captures.
"""
from pathlib import Path
import argparse,json,time
import numpy as np
from scipy.spatial import QhullError
B=Path('experiments/student_hall_lidar');O=Path('experiments/student_hall_refinement_20261027');HOLD=[5,13,21,29,37,45]
def dump(path,obj):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj,indent=2))
def cameras():
    rots=[]
    for forward in np.r_[np.eye(3),-np.eye(3)]:
        up=np.array([0.,-1,0]) if abs(forward[1])<.9 else np.array([0.,0,1])
        right=np.cross(forward,up);right/=np.linalg.norm(right);down=np.cross(forward,right);R=np.column_stack((right,down,forward));assert np.linalg.det(R)>.999
        rots.append(R)
    width=512;f=width/2/np.tan(np.deg2rad(100)/2);K=np.array([[f,0,255.5],[0,f,255.5],[0,0,1]])
    E=np.load(B/'calibration.npz')['cam2lidar'];origin=np.linalg.inv(E)[:3,3]
    mats=np.repeat(np.eye(4)[None],6,axis=0);mats[:,:3,:3]=rots;mats[:,:3,3]=origin
    return K,mats
def prepare():
    from reconstruct_student_hall_lidar import dense_depth
    K,M=cameras();folder=O/'cube_depth';folder.mkdir(exist_ok=True);np.savez(O/'cube_calibration.npz',K=K,cam_from_cube=M,shape=[512,512]);rows=[]
    for i in range(48):
        path=folder/f'{i:06d}.npz'
        if path.exists():rows.append(json.loads(path.with_suffix('.json').read_text()));continue
        x=np.load(B/'frames'/f'{i:06d}.npz')['camera'];x=x[np.linalg.norm(x,axis=1)>3.];depths=[];counts=[];start=time.time()
        for j,T in enumerate(M):
            face=(x-T[:3,3])@T[:3,:3];valid=(face[:,2]>.6)&(face[:,2]<45);z=face[valid];uv=z[:,:2]/z[:,2:]*K[0,0]+K[:2,2]
            good=(uv[:,0]>1)&(uv[:,0]<510)&(uv[:,1]>1)&(uv[:,1]<510)
            if good.sum()<20:d=np.zeros((512,512),np.float32);count=0
            else:
                try:s,d,_,_=dense_depth(face,K,(512,512));count=int((s>0).sum())
                except QhullError:d=np.zeros((512,512),np.float32);count=0;print('QHULL_EMPTY',i,j,flush=True)
            depths.append(d);counts.append(count)
        depth=np.asarray(depths);np.savez_compressed(path,depth=depth)
        row=dict(frame=i,seconds=time.time()-start,coverage=[float(np.mean(d>0)) for d in depth],sparse_returns=counts)
        dump(path.with_suffix('.json'),row);rows.append(row);print('CUBE_DEPTH',json.dumps(row),flush=True)
    dump(O/'cube_preparation.json',dict(views=rows,description='Six 100-degree virtual cameras centred at physical LiDAR origin; 512px. Same per-scan raw measured returns, conservative reciprocal-depth triangle interpolation. Camera pose transforms use existing metre calibration.'))
def integrate(voxel,ids):
    import open3d as o
    K,M=cameras();P=np.load(B/'fused/selected_poses.npz')['poses'];intr=o.camera.PinholeCameraIntrinsic(512,512,K[0,0],K[1,1],K[0,2],K[1,2])
    volume=o.pipelines.integration.ScalableTSDFVolume(voxel,voxel*4,o.pipelines.integration.TSDFVolumeColorType.NoColor)
    blank=o.geometry.Image(np.full((512,512,3),135,np.uint8))
    for i in ids:
        depth=np.load(O/'cube_depth'/f'{i:06d}.npz')['depth']
        for j in range(6):
            if (depth[j]>0).sum()<50:continue
            rgbd=o.geometry.RGBDImage.create_from_color_and_depth(blank,o.geometry.Image(depth[j].copy()),depth_scale=1,depth_trunc=38,convert_rgb_to_intensity=False)
            volume.integrate(rgbd,intr,np.linalg.inv(P[i]@M[j]))
    mesh=volume.extract_triangle_mesh();mesh.remove_degenerate_triangles();mesh.remove_duplicated_triangles();mesh.remove_unreferenced_vertices()
    cluster,counts,_=mesh.cluster_connected_triangles();counts=np.asarray(counts)
    if len(counts):mesh.remove_triangles_by_mask(counts[np.asarray(cluster)]<30)
    mesh.remove_unreferenced_vertices();mesh.compute_vertex_normals();return mesh
def raw_evaluate(mesh,poses):
    import open3d as o
    scene=o.t.geometry.RaycastingScene();scene.add_triangles(o.t.geometry.TriangleMesh.from_legacy(mesh));K=np.load(B/'fused/selected_poses.npz')['K'];origin=np.linalg.inv(np.load(B/'calibration.npz')['cam2lidar'])[:3,3];rows=[]
    for i in HOLD:
        x=np.load(B/'frames'/f'{i:06d}.npz')['camera'][::4];rel=x-origin;distance=np.linalg.norm(rel,axis=1);ok=(np.linalg.norm(x,axis=1)>3)&(distance<35);x=x[ok];rel=rel[ok];distance=distance[ok]
        unit=rel/distance[:,None];world_dir=unit@poses[i,:3,:3].T;world_origin=origin@poses[i,:3,:3].T+poses[i,:3,3]
        rays=np.column_stack((np.broadcast_to(world_origin,world_dir.shape),world_dir)).astype(np.float32);pred=scene.cast_rays(o.core.Tensor(rays))['t_hit'].numpy();has=np.isfinite(pred)
        err=np.minimum(np.abs(pred-distance),1);err[~has]=1
        uv=x[:,:2]/np.maximum(x[:,2:],.01)*[K[0,0],K[1,1]]+K[:2,2];front=(x[:,2]>.8)&(uv[:,0]>=0)&(uv[:,0]<960)&(uv[:,1]>=0)&(uv[:,1]<720)
        row=dict(frame=i)
        for name,mask in [('lidar360',np.ones(len(x),bool)),('camera_fov',front)]:
            row[name]=dict(samples=int(mask.sum()),coverage=float(has[mask].mean()),clipped_mae_cm=float(err[mask].mean()*100),fraction_within_10cm=float(np.mean(has[mask]&(np.abs(pred[mask]-distance[mask])<.1))))
        rows.append(row)
    hall=[r['camera_fov']['clipped_mae_cm'] for r in rows if r['frame']>=20]
    return dict(views=rows,hall_raw_ray_mae_cm=float(np.mean(hall)),description='Measured LiDAR rays from physical LiDAR origin, norm>3m, range<35m, every fourth return. Missing prediction and errors above 1m penalized at 1m. Camera-FOV subset for student hall. No interpolated target depth. Poses use all scans, so still a consistency diagnostic.')
def validation(voxel):
    import open3d as o
    from mesh_student_hall_lidar import evaluate
    folder=O/'surfaces';folder.mkdir(exist_ok=True);start=time.time();P=np.load(B/'fused/selected_poses.npz');mesh=integrate(voxel,[i for i in range(48) if i not in HOLD])
    row=evaluate(mesh,P['poses'],P['K'],HOLD);row.update(voxel_m=voxel,triangles=len(mesh.triangles),vertices=len(mesh.vertices),seconds=time.time()-start)
    row['student_hall_mae_cm']=float(np.mean([r['clipped_mae_cm'] for r in row['views'] if r['frame']>=20]));row['raw_rays']=raw_evaluate(mesh,P['poses'])
    path=folder/f'cube_tsdf_{voxel:.2f}_validation';o.io.write_triangle_mesh(str(path)+'.ply',mesh);dump(Path(str(path)+'.json'),row);print('CUBE_VALIDATION',json.dumps(row),flush=True)
def baseline_raw():
    import open3d as o
    mesh=o.io.read_triangle_mesh(str(B/'fused/validation_poisson_11.ply'));P=np.load(B/'fused/selected_poses.npz')['poses'];row=raw_evaluate(mesh,P);dump(O/'baseline_raw_ray_validation.json',row);print('BASELINE_RAW',json.dumps(row),flush=True)
def final(voxel):
    import open3d as o
    from refine_student_hall_surface import color
    mesh=integrate(voxel,range(48));z=np.load(B/'fused/selected_poses.npz');stats=color(mesh,z['poses'],z['K']);folder=O/'surfaces';folder.mkdir(exist_ok=True)
    o.io.write_triangle_mesh(str(folder/f'cube_tsdf_{voxel:.2f}_final.ply'),mesh);preview=mesh.simplify_quadric_decimation(180000);preview.compute_vertex_normals();o.io.write_triangle_mesh(str(folder/f'cube_tsdf_{voxel:.2f}_preview.ply'),preview)
    stats.update(voxel_m=voxel,triangles=len(mesh.triangles));dump(folder/f'cube_tsdf_{voxel:.2f}_final.json',stats);print('CUBE_FINAL',json.dumps(stats),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','validation','baseline_raw','final']);p.add_argument('--voxel',type=float,default=.06);a=p.parse_args()
    globals()[a.stage](a.voxel) if a.stage in ['validation','final'] else globals()[a.stage]()
