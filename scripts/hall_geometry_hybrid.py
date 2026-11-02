#!/usr/bin/env python3
"""Conservative augmentation of measured geometry with photo-triangulated points."""
from pathlib import Path
import argparse,json,time
import numpy as np,open3d as o
from scipy.spatial import cKDTree
from hall_geometry_surface import assess
from hall_followup_surface import HOLD
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'experiments/student_hall_geometry_20261031';B=ROOT/'experiments/student_hall_lidar'
def main(a):
    start=time.time();P=np.load(B/'fused/selected_poses.npz')['poses'];ids=list(range(48)) if a.final else [i for i in range(48) if i not in HOLD];pieces=[]
    for i in ids:
        x=np.load(B/'frames'/f'{i:06d}.npz')['camera'][::2];x=x[np.linalg.norm(x,axis=1)>3];p=o.geometry.PointCloud(o.utility.Vector3dVector(x)).voxel_down_sample(.12);p.estimate_normals(o.geometry.KDTreeSearchParamHybrid(radius=.4,max_nn=30));p.orient_normals_towards_camera_location([0,0,0]);p.transform(P[i]);pieces.append(p)
    p=o.geometry.PointCloud()
    for piece in pieces:p+=piece
    p=p.voxel_down_sample(.1);p,_=p.remove_statistical_outlier(nb_neighbors=20,std_ratio=2);base_count=len(p.points);tree=cKDTree(np.asarray(p.points));photo=o.io.read_point_cloud(str(OUT/f'mvs_{a.variant}_validation/photo_dense_metric.ply'));photo=photo.voxel_down_sample(.025);x=np.asarray(photo.points);dist,nn=tree.query(x,workers=4);normal=np.asarray(photo.normals);targetnormal=np.asarray(p.normals)[nn];compatible=np.abs(np.sum(normal*targetnormal,axis=1))>.85;keep=(dist<a.support)&compatible
    photo=photo.select_by_index(np.flatnonzero(keep));normal=np.asarray(photo.normals);targetnormal=targetnormal[keep];normal[np.sum(normal*targetnormal,axis=1)<0]*=-1;photo.normals=o.utility.Vector3dVector(normal)
    # Keep the original measured cloud exactly; add only photo points close to
    # compatible measured surfaces. No unsupported remote walls or sky closure.
    p+=photo;name=f'hybrid_{a.variant}_support{a.support:.2f}'+('_final' if a.final else '_validation');folder=OUT/name;folder.mkdir(exist_ok=True)
    print('HYBRID_INPUT',base_count,len(photo.points),flush=True);mesh,density=o.geometry.TriangleMesh.create_from_point_cloud_poisson(p,depth=11,n_threads=1,linear_fit=False);xyz=np.asarray(p.points);distance,_=cKDTree(xyz).query(np.asarray(mesh.vertices),workers=4);mesh.remove_vertices_by_mask((distance>.25)|(np.asarray(density)<np.quantile(density,.015)));mesh.remove_degenerate_triangles();mesh.remove_unreferenced_vertices();clusters,counts,_=mesh.cluster_connected_triangles();counts=np.array(counts)
    if len(counts):mesh.remove_triangles_by_mask(counts[np.array(clusters)]<60)
    mesh.remove_unreferenced_vertices();mesh.compute_vertex_normals();o.io.write_triangle_mesh(str(folder/'surface.ply'),mesh)
    row=dict(name=name,measured_input_points=base_count,photo_input_points=len(photo.points),photo_support_distance_m=a.support,vertices=len(mesh.vertices),triangles=len(mesh.triangles),seconds=time.time()-start,geometry_source='Measured LiDAR plus separately triangulated dense photo points, with distance and normal checks against training LiDAR. Not photograph-only.',caveat='MVS photo points use validation-filtered images, SfM poses use all images, and metric alignment uses training LiDAR. Repeated candidate evaluation is not independent final testing.')
    if not a.final:row['validation']=assess(mesh)
    else:
        from refine_student_hall_surface import color
        c=np.load(B/'fused/selected_poses.npz');row['color']=color(mesh,c['poses'],c['K']);o.io.write_triangle_mesh(str(folder/'surface_colored.ply'),mesh)
    (folder/'surface_report.json').write_text(json.dumps(row,indent=2));print('HYBRID_RESULT',json.dumps(row),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--support',type=float,default=.10);p.add_argument('--variant',default='fixed_high');p.add_argument('--final',action='store_true');main(p.parse_args())
