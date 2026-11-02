#!/usr/bin/env python3
"""TSDF sweep + withheld-scan surface checks in Python 3.12 / Open3D 0.19."""
from pathlib import Path
import argparse,json,time
import numpy as np
import open3d as o

ROOT=Path('experiments/student_hall_lidar/fused')

def integrate(poses,K,voxel,ids):
    intrinsic=o.camera.PinholeCameraIntrinsic(960,720,K[0,0],K[1,1],K[0,2],K[1,2])
    vol=o.pipelines.integration.ScalableTSDFVolume(voxel,voxel*4,o.pipelines.integration.TSDFVolumeColorType.RGB8)
    for i in ids:
        z=np.load(ROOT/f'rgbd_{i:02d}.npz')
        rgbd=o.geometry.RGBDImage.create_from_color_and_depth(o.geometry.Image(z['rgb'].copy()),o.geometry.Image(z['depth'].copy()),
            depth_scale=1.,depth_trunc=38.,convert_rgb_to_intensity=False)
        vol.integrate(rgbd,intrinsic,np.linalg.inv(poses[i]))
    mesh=vol.extract_triangle_mesh();mesh.remove_degenerate_triangles();mesh.remove_duplicated_triangles();mesh.remove_unreferenced_vertices()
    # Remove isolated fragments while retaining independently observed building surfaces.
    cluster,counts,_=mesh.cluster_connected_triangles();counts=np.asarray(counts)
    if len(counts):mesh.remove_triangles_by_mask(counts[np.asarray(cluster)]<30)
    mesh.remove_unreferenced_vertices();mesh.compute_vertex_normals()
    return mesh

def evaluate(mesh,poses,K,ids):
    scene=o.t.geometry.RaycastingScene();scene.add_triangles(o.t.geometry.TriangleMesh.from_legacy(mesh))
    rows=[];errors=[]
    for i in ids:
        depth=np.load(ROOT/f'rgbd_{i:02d}.npz')['depth']
        yy,xx=np.mgrid[0:720:4,0:960:4];z=depth[yy,xx];ok=(z>.8)&(z<35)
        xx,yy,z=xx[ok],yy[ok],z[ok]
        direction=np.column_stack(((xx-K[0,2])/K[0,0],(yy-K[1,2])/K[1,1],np.ones(len(xx))))
        direction=direction@poses[i,:3,:3].T;origin=np.repeat(poses[i:i+1,:3,3],len(xx),axis=0)
        rays=np.column_stack((origin,direction)).astype(np.float32)
        pred=scene.cast_rays(o.core.Tensor(rays))['t_hit'].numpy();has=np.isfinite(pred)
        err=np.minimum(np.abs(pred-z),1.);err[~has]=1.
        errors.extend(err);rows.append({'frame':int(i),'coverage':float(has.mean()),'clipped_mae_cm':float(err.mean()*100)})
    return {'clipped_mae_cm':float(np.mean(errors)*100),'samples':len(errors),'views':rows}

def main():
    p=argparse.ArgumentParser();p.add_argument('--skip-sweep',action='store_true');a=p.parse_args()
    q=np.load(ROOT/'selected_poses.npz');poses,K=q['poses'],q['K']
    heldout=[5,13,21,29,37,45];train=[i for i in range(48) if i not in heldout];rows=[]
    if not a.skip_sweep:
        for voxel in [.06,.10,.15]:
            t=time.time();mesh=integrate(poses,K,voxel,train);r=evaluate(mesh,poses,K,heldout)
            r.update(voxel_m=voxel,triangles=len(mesh.triangles),elapsed_s=time.time()-t);rows.append(r)
            print('mesh_validation',json.dumps(r),flush=True)
            o.io.write_triangle_mesh(str(ROOT/f'validation_tsdf_{voxel:.2f}.ply'),mesh)
        voxel=min(rows,key=lambda r:r['clipped_mae_cm'])['voxel_m']
        (ROOT/'mesh_validation.json').write_text(json.dumps({'candidates':rows,'selected_voxel_m':voxel,
            'note':'Six complete depth frames excluded from validation-mesh integration. Poses were estimated using all frames. Errors clipped at 1m; missing surfaces penalized at 1m. Depths originate from LiDAR interpolation, not survey ground truth.'},indent=2))
    else:voxel=json.loads((ROOT/'mesh_validation.json').read_text())['selected_voxel_m']
    final=integrate(poses,K,voxel,range(48));o.io.write_triangle_mesh(str(ROOT/'student_hall_tsdf.ply'),final)
    preview=final.simplify_quadric_decimation(150000);preview.compute_vertex_normals()
    o.io.write_triangle_mesh(str(ROOT/'student_hall_tsdf_preview.ply'),preview)
    summary=json.loads((ROOT/'summary.json').read_text());summary.update(mesh_triangles=len(final.triangles),mesh_vertices=len(final.vertices),mesh_voxel_m=voxel)
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2))
    print('final',summary,flush=True)

if __name__=='__main__':main()
