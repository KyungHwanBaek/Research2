#!/usr/bin/env python3
"""Alternative full-LiDAR mesh with support trimming and withheld-scan evaluation."""
from pathlib import Path
import json
import numpy as np
import open3d as o
from scipy.spatial import cKDTree
from mesh_student_hall_lidar import evaluate

ROOT=Path('experiments/student_hall_lidar');OUT=ROOT/'fused'

def build(ids,poses,depth):
    pieces=[]
    for i in ids:
        x=np.load(ROOT/'frames'/f'{i:06d}.npz')['camera'][::2]
        # Near-range self returns otherwise form a false tube along the sensor path.
        x=x[np.linalg.norm(x,axis=1)>3.]
        p=o.geometry.PointCloud(o.utility.Vector3dVector(x.astype(float))).voxel_down_sample(.12)
        p.estimate_normals(o.geometry.KDTreeSearchParamHybrid(radius=.4,max_nn=30))
        p.orient_normals_towards_camera_location([0,0,0]);p.transform(poses[i]);pieces.append(p)
    p=o.geometry.PointCloud()
    for part in pieces:p+=part
    p=p.voxel_down_sample(.10)
    p,keep=p.remove_statistical_outlier(nb_neighbors=20,std_ratio=2.)
    mesh,density=o.geometry.TriangleMesh.create_from_point_cloud_poisson(p,depth=depth,n_threads=1,linear_fit=False)
    xyz=np.asarray(p.points);v=np.asarray(mesh.vertices)
    distance,_=cKDTree(xyz).query(v,workers=4)
    # Reject inferred closure far from actual returns; do not close entire unseen rooms.
    bad=(distance>.25)|(np.asarray(density)<np.quantile(density,.015))
    mesh.remove_vertices_by_mask(bad);mesh.remove_degenerate_triangles();mesh.remove_unreferenced_vertices()
    clusters,counts,_=mesh.cluster_connected_triangles();counts=np.asarray(counts)
    if len(counts):mesh.remove_triangles_by_mask(counts[np.asarray(clusters)]<60)
    mesh.remove_unreferenced_vertices();mesh.compute_vertex_normals()
    return mesh

def main():
    z=np.load(OUT/'selected_poses.npz');poses,K=z['poses'],z['K'];heldout=[5,13,21,29,37,45]
    train=[i for i in range(48) if i not in heldout];rows=[]
    for d in [10,11]:
        mesh=build(train,poses,d);r=evaluate(mesh,poses,K,heldout);r.update(depth=d,triangles=len(mesh.triangles))
        r['student_hall_mae_cm']=float(np.mean([v['clipped_mae_cm'] for v in r['views'] if v['frame']>=20]))
        rows.append(r);print('poisson_validation',json.dumps(r),flush=True)
        o.io.write_triangle_mesh(str(OUT/f'validation_poisson_{d}.ply'),mesh)
    d=min(rows,key=lambda r:r['student_hall_mae_cm'])['depth'];mesh=build(range(48),poses,d)
    p=o.io.read_point_cloud(str(OUT/'lidar_full_map.ply'));x=np.asarray(p.points);v=np.asarray(mesh.vertices)
    _,nn=cKDTree(x).query(v,workers=4);mesh.vertex_colors=o.utility.Vector3dVector(np.asarray(p.colors)[nn])
    o.io.write_triangle_mesh(str(OUT/'student_hall_poisson.ply'),mesh)
    preview=mesh.simplify_quadric_decimation(180000);preview.compute_vertex_normals()
    o.io.write_triangle_mesh(str(OUT/'student_hall_poisson_preview.ply'),preview)
    report={'candidates':rows,'selected_depth':d,'final_triangles':len(mesh.triangles),
         'note':'Full 360-degree LiDAR Poisson mesh; faces farther than 25cm from measured support removed. Six complete scans withheld from candidate mesh integration; all-scan poses reused.'}
    (OUT/'poisson_validation.json').write_text(json.dumps(report,indent=2))
    summary=json.loads((OUT/'summary.json').read_text());summary.update(selected_mesh='poisson',
        mesh_triangles=len(mesh.triangles),mesh_vertices=len(mesh.vertices),poisson_depth=d,minimum_lidar_range_m=3.,
        tsdf_mesh_triangles=len(o.io.read_triangle_mesh(str(OUT/'student_hall_tsdf.ply')).triangles))
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    print('poisson_final',report,flush=True)

if __name__=='__main__':main()
