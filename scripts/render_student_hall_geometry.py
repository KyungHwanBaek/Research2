#!/usr/bin/env python3
"""Headless image evidence of reconstructed geometry, without a browser or GUI."""
from pathlib import Path
import numpy as np
import open3d as o
from PIL import Image,ImageDraw

R=Path('experiments/student_hall_lidar/fused');OUT=R.parent

def render(mesh,P,K,w=960,h=720):
    scene=o.t.geometry.RaycastingScene();scene.add_triangles(o.t.geometry.TriangleMesh.from_legacy(mesh))
    yy,xx=np.indices((h,w));d=np.stack(((xx-K[0,2])/K[0,0],(yy-K[1,2])/K[1,1],np.ones_like(xx)),axis=-1)
    rays=np.concatenate((np.broadcast_to(P[:3,3],d.shape),d@P[:3,:3].T),axis=-1).astype(np.float32)
    hit=scene.cast_rays(o.core.Tensor(rays));prim=hit['primitive_ids'].numpy();b=hit['primitive_uvs'].numpy()
    valid=np.isfinite(hit['t_hit'].numpy());tri=np.asarray(mesh.triangles);colors=np.asarray(mesh.vertex_colors)
    im=np.full((h,w,3),[236,240,237],np.uint8)
    ids=prim[valid];uv=b[valid];weight=np.column_stack((1-uv.sum(1),uv))
    c=np.sum(colors[tri[ids]]*weight[:,:,None],axis=1)
    normals=np.asarray(mesh.triangle_normals)[ids];light=.65+.35*np.abs(normals@np.array([.3,-.6,-.74]))
    im[valid]=np.uint8(np.clip(c*light[:,None],0,1)*255)
    return im

def main():
    z=np.load(R/'selected_poses.npz');ps,K=z['poses'],z['K'];panels=[]
    for name in ['poisson','tsdf']:
        mesh=o.io.read_triangle_mesh(str(R/f'student_hall_{name}.ply'));mesh.compute_triangle_normals()
        for i in [26,38,47]:
            im=render(mesh,ps[i],K);path=OUT/f'{name}_view_{i}.jpg';Image.fromarray(im).save(path)
            thumb=Image.fromarray(im).resize((480,360));draw=ImageDraw.Draw(thumb);draw.text((10,10),f'{name} | camera {i}',fill='black',stroke_width=2,stroke_fill='white');panels.append(thumb)
    sheet=Image.new('RGB',(1440,720),'white')
    for j,p in enumerate(panels):sheet.paste(p,((j%3)*480,(j//3)*360))
    sheet.save(OUT/'geometry_contact_sheet.jpg')

if __name__=='__main__':main()
