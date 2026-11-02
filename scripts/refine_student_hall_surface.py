#!/usr/bin/env python3
"""Finer measured surfaces / stereo extension, whole-scan validation and color.

Run under Open3D 0.19, not 0.20. Supports separate validation and final stages.
"""
from pathlib import Path
import argparse,json,time
import numpy as np
import open3d as o
from scipy.spatial import cKDTree
from mesh_student_hall_lidar import evaluate

BASE=Path('experiments/student_hall_lidar');OUT=Path('experiments/student_hall_refinement_20261026')
HOLD=[5,13,21,29,37,45]

def build(ids,poses,voxel,depth,hybrid=False):
    parts=[]
    for i in ids:
        x=np.load(BASE/'frames'/f'{i:06d}.npz')['camera'][::2];x=x[np.linalg.norm(x,axis=1)>3.]
        p=o.geometry.PointCloud(o.utility.Vector3dVector(x.astype(float))).voxel_down_sample(voxel)
        p.estimate_normals(o.geometry.KDTreeSearchParamHybrid(radius=voxel*4,max_nn=30));p.orient_normals_towards_camera_location([0,0,0]);p.transform(poses[i]);parts.append(p)
    if hybrid:
        q=np.load(OUT/'rectified_poses.npz');K=q['K'];ps=q['poses']
        for i in ids:
            z=np.load(OUT/'hybrid'/f'{i:06d}.npz');d=z['verified'];y,x=np.mgrid[0:720:3,0:960:3];dep=d[y,x];ok=dep>0;x=x[ok];y=y[ok];dep=dep[ok]
            xyz=np.column_stack(((x-K[0,2])/K[0,0]*dep,(y-K[1,2])/K[1,1]*dep,dep))
            if len(xyz)<30:continue
            p=o.geometry.PointCloud(o.utility.Vector3dVector(xyz)).voxel_down_sample(voxel)
            p.estimate_normals(o.geometry.KDTreeSearchParamHybrid(radius=voxel*4,max_nn=30));p.orient_normals_towards_camera_location([0,0,0]);p.transform(ps[i]);parts.append(p)
    p=o.geometry.PointCloud()
    for pp in parts:p+=pp
    p=p.voxel_down_sample(voxel);p,_=p.remove_statistical_outlier(nb_neighbors=20,std_ratio=2.)
    print('poisson_input',len(p.points),'voxel',voxel,'depth',depth,'hybrid',hybrid,flush=True)
    mesh,density=o.geometry.TriangleMesh.create_from_point_cloud_poisson(p,depth=depth,n_threads=1,linear_fit=False)
    dist,_=cKDTree(np.asarray(p.points)).query(np.asarray(mesh.vertices),workers=4)
    bad=(dist>.20)|(np.asarray(density)<np.quantile(density,.01));mesh.remove_vertices_by_mask(bad)
    mesh.remove_degenerate_triangles();mesh.remove_unreferenced_vertices()
    clusters,counts,_=mesh.cluster_connected_triangles();counts=np.asarray(counts)
    if len(counts):mesh.remove_triangles_by_mask(counts[np.asarray(clusters)]<60)
    mesh.remove_unreferenced_vertices();mesh.compute_vertex_normals()
    return mesh

def color(mesh,poses,K):
    import cv2
    # Render mesh visibility for each original photo, then blend measured RGB
    # with angle / distance / edge weighting. Unseen faces remain neutral.
    scene=o.t.geometry.RaycastingScene();scene.add_triangles(o.t.geometry.TriangleMesh.from_legacy(mesh))
    v=np.asarray(mesh.vertices);n=np.asarray(mesh.vertex_normals);acc=np.zeros_like(v);weight=np.zeros(len(v));best=np.zeros(len(v))
    for i in range(48):
        T=np.linalg.inv(poses[i]);x=v@T[:3,:3].T+T[:3,3];uv=x[:,:2]/np.maximum(x[:,2:],.001)*[K[0,0],K[1,1]]+K[:2,2]
        ij=np.rint(uv).astype(int);ok=(x[:,2]>1)&(x[:,2]<40)&(ij[:,0]>=2)&(ij[:,0]<958)&(ij[:,1]>=2)&(ij[:,1]<718)
        ids=np.flatnonzero(ok)
        # Exact rays to projected vertices avoid low-resolution visibility errors.
        direction=v[ids]-poses[i,:3,3];dist=np.linalg.norm(direction,axis=1);direction/=dist[:,None]
        rays=np.column_stack((np.broadcast_to(poses[i,:3,3],direction.shape),direction)).astype(np.float32)
        hit=scene.cast_rays(o.core.Tensor(rays))['t_hit'].numpy();visible=np.abs(hit-dist)<(.04+.003*dist)
        ids=ids[visible];dist=dist[visible];direction=direction[visible];u,w=ij[ids].T
        angle=np.abs(np.sum(n[ids]*direction,axis=1));edge=np.minimum.reduce((u,959-u,w,719-w))/120
        score=angle**2*np.clip(edge,0,1)/(1+(dist/12)**2)
        im=cv2.imread(str(BASE/'images'/f'{i:06d}.png'))[:,:,::-1]/255
        # Keep only similarly good views to prevent oblique colors smearing detail.
        accept=score>best[ids]*.5;ids=ids[accept];score=score[accept];u=u[accept];w=w[accept]
        reset=score>best[ids]*2;acc[ids[reset]]=0;weight[ids[reset]]=0
        acc[ids]+=im[w,u]*score[:,None];weight[ids]+=score;best[ids]=np.maximum(best[ids],score)
    colors=np.full_like(v,.53);observed=weight>1e-6;colors[observed]=acc[observed]/weight[observed,None]
    mesh.vertex_colors=o.utility.Vector3dVector(colors)
    return dict(vertices=len(v),photo_colored_vertices=int(observed.sum()),photo_colored_fraction=float(observed.mean()),
        method='Visibility-tested actual photograph projection; angle/distance/edge weighted blending; unobserved vertices neutral gray.')

def main(a):
    z=np.load(BASE/'fused/selected_poses.npz');P,K=z['poses'],z['K'];name='baseline_recolored' if a.recolor else f'{a.mode}_v{a.voxel:.3f}_d{a.depth}';folder=OUT/'surfaces';folder.mkdir(exist_ok=True)
    if a.poses=='global':P=np.load(OUT/'global_poses.npz')['poses'];name='global_'+name
    if a.baseline_build:name=a.poses+'_baseline_parameters'
    ids=range(48) if a.final else [i for i in range(48) if i not in HOLD]
    t=time.time()
    if a.recolor:mesh=o.io.read_triangle_mesh(str(BASE/'fused/student_hall_poisson.ply'))
    elif a.baseline_build:
        from poisson_student_hall import build as original_build
        mesh=original_build(ids,P,11)
    else:mesh=build(ids,P,a.voxel,a.depth,a.mode=='hybrid')
    mesh.compute_vertex_normals()
    if not a.final:
        report=evaluate(mesh,P,K,HOLD);report.update(name=name,voxel_m=a.voxel,depth=a.depth,mode=a.mode,triangles=len(mesh.triangles),elapsed_s=time.time()-t)
        report['student_hall_mae_cm']=float(np.mean([r['clipped_mae_cm'] for r in report['views'] if r['frame']>=20]))
        (folder/f'{name}_validation.json').write_text(json.dumps(report,indent=2));print('surface_validation',json.dumps(report),flush=True)
        o.io.write_triangle_mesh(str(folder/f'{name}_validation.ply'),mesh)
    else:
        report=color(mesh,P,K);report.update(name=name,triangles=len(mesh.triangles),elapsed_s=time.time()-t)
        o.io.write_triangle_mesh(str(folder/f'{name}_final.ply'),mesh)
        preview=mesh.simplify_quadric_decimation(180000);preview.compute_vertex_normals();o.io.write_triangle_mesh(str(folder/f'{name}_preview.ply'),preview)
        (folder/f'{name}_final.json').write_text(json.dumps(report,indent=2));print('surface_final',json.dumps(report),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['lidar','hybrid'],default='lidar');p.add_argument('--voxel',type=float,default=.06);p.add_argument('--depth',type=int,default=11);p.add_argument('--final',action='store_true');p.add_argument('--recolor',action='store_true');p.add_argument('--poses',choices=['baseline','global'],default='baseline');p.add_argument('--baseline-build',action='store_true');main(p.parse_args())
