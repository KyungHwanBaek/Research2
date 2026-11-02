#!/usr/bin/env python3
"""TSDF mesh generation in the separate Python 3.12 / Open3D 0.19 environment."""
from pathlib import Path
import json
import argparse
import cv2
import numpy as np
import open3d as o3d
from improve_stereo_reconstruction import Stereo


def verify_tsdf_backend():
    k=o3d.camera.PinholeCameraIntrinsic(64,48,55.,55.,32.,24.)
    rgb=o3d.geometry.Image(np.full((48,64,3),128,np.uint8))
    depth=o3d.geometry.Image(np.full((48,64),2.,np.float32))
    rgbd=o3d.geometry.RGBDImage.create_from_color_and_depth(rgb,depth,depth_scale=1.,depth_trunc=5.,convert_rgb_to_intensity=False)
    v=o3d.pipelines.integration.ScalableTSDFVolume(.05,.2,o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    v.integrate(rgbd,k,np.eye(4))
    assert len(v.extract_triangle_mesh().triangles)>0, f'TSDF backend {o3d.__version__} failed a known plane test'


def main():
    p=argparse.ArgumentParser();p.add_argument('--poses',default='experiments/stereo_improved_0905/optimized_poses.npz')
    p.add_argument('--output',default='experiments/stereo_improved_0905/fused');a=p.parse_args()
    verify_tsdf_backend();cv2.setNumThreads(4)
    s=Stereo('data/extracted/sampled_540p/09-05-17-34-36','experiments/stereo_improved_0905/depth')
    z=np.load(a.poses);indices=z['indices'];poses=z['poses'];segments=z['components']
    out=Path(a.output);stats=json.loads((out/'summary.json').read_text())
    intrinsic=o3d.camera.PinholeCameraIntrinsic(s.size[0],s.size[1],s.rectK[0,0],s.rectK[1,1],s.rectK[0,2],s.rectK[1,2])
    for seg in np.unique(segments):
        volume=o3d.pipelines.integration.ScalableTSDFVolume(.06,.24,o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
        ids=np.flatnonzero(segments==seg)[::2]
        for j,k in enumerate(ids):
            i=int(indices[k]);depth=s.combined_depth(i)
            yy,xx=np.mgrid[0:depth.shape[0]:2,0:depth.shape[1]:2]
            uv=np.column_stack((xx.ravel(),yy.ravel())).astype(np.float32)
            xyz=s.points(depth,uv);valid=(xyz[:,2]>0)&(xyz[:,2]<28)
            xyz,uv=xyz[valid],uv[valid];world=xyz@poses[k,:3,:3].T+poses[k,:3,3]
            supported=np.zeros(len(xyz),bool)
            for delta in [-2,2]:
                n=k+delta
                if n<0 or n>=len(indices) or segments[n]!=seg:continue
                dn=s.combined_depth(int(indices[n]))
                cam=(world-poses[n,:3,3])@poses[n,:3,:3]
                projected=cam[:,:2]/np.maximum(cam[:,2:],1e-6)*np.array([s.rectK[0,0],s.rectK[1,1]])+s.rectK[:2,2]
                xy=np.rint(projected).astype(int)
                valid=(cam[:,2]>.5)&(xy[:,0]>=0)&(xy[:,0]<s.size[0])&(xy[:,1]>=0)&(xy[:,1]<s.size[1])
                kk=np.flatnonzero(valid);observed=dn[xy[kk,1],xy[kk,0]]
                supported[kk]|=(observed>0)&(np.abs(observed-cam[kk,2])<.055*observed)
            mask=np.zeros_like(depth,np.uint8);mask[uv[supported,1].astype(int),uv[supported,0].astype(int)]=1
            mask=cv2.dilate(mask,np.ones((3,3),np.uint8))>0
            clean=np.where(mask,depth,0).astype(np.float32)
            color=np.ascontiguousarray(s.image(i)[:,:,::-1])
            rgbd=o3d.geometry.RGBDImage.create_from_color_and_depth(o3d.geometry.Image(color),o3d.geometry.Image(clean),
                    depth_scale=1.,depth_trunc=28.,convert_rgb_to_intensity=False)
            volume.integrate(rgbd,intrinsic,np.linalg.inv(poses[k]))
            if j%30==0:print('surface_integrate','segment',int(seg),'frame',j,'/',len(ids),flush=True)
        mesh=volume.extract_triangle_mesh();mesh.compute_vertex_normals()
        print('raw_surface',int(seg),len(mesh.vertices),len(mesh.triangles),flush=True)
        mesh.remove_degenerate_triangles();mesh.remove_duplicated_triangles();mesh.remove_unreferenced_vertices()
        labels,counts,_=mesh.cluster_connected_triangles()
        mesh.remove_triangles_by_mask(np.asarray(counts)[np.asarray(labels)]<150)
        mesh.remove_unreferenced_vertices()
        assert len(mesh.triangles)>0,'No supported surface remains'
        stem=out/f'stereo_rgb_nir_segment{int(seg)}'
        assert o3d.io.write_triangle_mesh(str(stem)+'_tsdf_mesh.ply',mesh)
        preview=mesh.simplify_quadric_decimation(180000);preview.compute_vertex_normals()
        assert o3d.io.write_triangle_mesh(str(stem)+'_tsdf_preview.ply',preview)
        report=next(r for r in stats if r['segment']==int(seg))
        report.update(mesh_vertices=len(mesh.vertices),mesh_triangles=len(mesh.triangles),mesh_backend=f'Open3D {o3d.__version__}',mesh_voxel_size_m=.06)
        stem.with_suffix('.json').write_text(json.dumps(report,indent=2))
        (out/'summary.json').write_text(json.dumps(stats,indent=2))
        print('surface_saved',stem,'triangles',len(mesh.triangles),flush=True)


if __name__=='__main__':main()
