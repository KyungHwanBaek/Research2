#!/usr/bin/env python3
"""Data-guided local planar correction of the existing Poisson surface.

Only near-planar, normal-compatible neighborhoods are moved; no hole filling.
Validation cloud excludes all six complete validation scans.
"""
from pathlib import Path
import argparse,json,time
import numpy as np
import open3d as o
from scipy.spatial import cKDTree
from mesh_student_hall_lidar import evaluate
from hall_followup_surface import raw_evaluate,HOLD
B=Path('experiments/student_hall_lidar');O=Path('experiments/student_hall_refinement_20261027')
def points(ids):
    P=np.load(B/'fused/selected_poses.npz')['poses'];parts=[]
    for i in ids:
        x=np.load(B/'frames'/f'{i:06d}.npz')['camera'][::2];x=x[np.linalg.norm(x,axis=1)>3];parts.append(x@P[i,:3,:3].T+P[i,:3,3])
    p=o.geometry.PointCloud(o.utility.Vector3dVector(np.concatenate(parts).astype(float))).voxel_down_sample(.08);p,_=p.remove_statistical_outlier(nb_neighbors=20,std_ratio=2);return np.asarray(p.points)
def refine(mesh,cloud,radius,limit,safe=False):
    mesh.compute_vertex_normals();v=np.asarray(mesh.vertices).copy();normal=np.asarray(mesh.vertex_normals).copy();tree=cKDTree(cloud);out=v.copy();moved=[]
    for start in range(0,len(v),20000):
        x=v[start:start+20000];d,j=tree.query(x,k=20,workers=3);p=cloud[j];w=np.exp(-(d/(radius*.55))**2);w[d>radius]=0;count=(w>0).sum(1);w/=np.maximum(w.sum(1,keepdims=True),1e-12)
        center=(p*w[:,:,None]).sum(1);q=p-center[:,None];cov=np.einsum('nki,nkj,nk->nij',q,q,w);eig,U=np.linalg.eigh(cov);n=U[:,:,0];shift=np.sum((x-center)*n,axis=1)
        planar=eig[:,0]/np.maximum(eig.sum(1),1e-10)<.06;compatible=np.abs(np.sum(n*normal[start:start+len(x)],axis=1))>.9
        good=(count>=8)&planar&compatible&(np.abs(shift)<limit)&(d[:,0]<.15)
        # A single conservative projection; no repeated smoothing/shrinking.
        out[start:start+len(x)][good]=x[good]-shift[good,None]*n[good];moved.extend(np.abs(shift[good]))
    reverted=set();flips_before=0
    if safe:
        faces=np.asarray(mesh.triangles);old=v[faces];old_normal=np.cross(old[:,1]-old[:,0],old[:,2]-old[:,0]);del old
        for iteration in range(40):
            new=out[faces];new_normal=np.cross(new[:,1]-new[:,0],new[:,2]-new[:,0]);bad=np.sum(old_normal*new_normal,axis=1)<0
            if iteration==0:flips_before=int(bad.sum())
            if not bad.any():break
            indices=np.unique(faces[bad]);reverted.update(indices.tolist());out[indices]=v[indices]
        else:raise RuntimeError('Orientation preservation did not converge')
    movement=np.linalg.norm(out-v,axis=1);moved=movement[movement>1e-10]
    mesh.vertices=o.utility.Vector3dVector(out);mesh.remove_degenerate_triangles();mesh.compute_vertex_normals()
    return mesh,dict(moved_vertices=len(moved),vertices=len(v),median_movement_cm=float(np.median(moved)*100) if len(moved) else 0,max_movement_cm=float(np.max(moved)*100) if len(moved) else 0,radius_m=radius,limit_m=limit,orientation_preservation=safe,initial_flipped_faces=flips_before,reverted_vertices=len(reverted))
def main(a):
    folder=O/'surfaces';folder.mkdir(exist_ok=True);P=np.load(B/'fused/selected_poses.npz');name=f'mls_r{a.radius:.2f}_l{a.limit:.2f}'+('_safe' if a.safe else '');start=time.time()
    if a.final:
        mesh=o.io.read_triangle_mesh(str(B/'fused/student_hall_poisson.ply'));ids=range(48)
    else:mesh=o.io.read_triangle_mesh(str(B/'fused/validation_poisson_11.ply'));ids=[i for i in range(48) if i not in HOLD]
    cloud=points(ids);mesh,report=refine(mesh,cloud,a.radius,a.limit,a.safe);report.update(name=name,train_cloud_points=len(cloud),triangles=len(mesh.triangles),seconds=time.time()-start)
    if not a.final:
        val=evaluate(mesh,P['poses'],P['K'],HOLD);report['validation']=val;report['student_hall_mae_cm']=float(np.mean([v['clipped_mae_cm'] for v in val['views'] if v['frame']>=20]));report['raw_rays']=raw_evaluate(mesh,P['poses']);suffix='validation'
    else:
        from refine_student_hall_surface import color
        report['color']=color(mesh,P['poses'],P['K']);suffix='final'
    o.io.write_triangle_mesh(str(folder/f'{name}_{suffix}.ply'),mesh);(folder/f'{name}_{suffix}.json').write_text(json.dumps(report,indent=2));print('MLS_SURFACE',json.dumps(report),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--radius',type=float,default=.25);p.add_argument('--limit',type=float,default=.05);p.add_argument('--final',action='store_true');p.add_argument('--safe',action='store_true');main(p.parse_args())
