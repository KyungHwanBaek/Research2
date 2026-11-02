#!/usr/bin/env python3
"""Fixed-camera before/after surface-color evidence (same geometry)."""
from pathlib import Path
import numpy as np
import open3d as o
from PIL import Image,ImageDraw
from render_student_hall_geometry import render
B=Path('experiments/student_hall_lidar');O=Path('experiments/student_hall_refinement_20261026')
def main():
    z=np.load(B/'fused/selected_poses.npz');panels=[]
    for tag,path in [('before',B/'fused/student_hall_poisson.ply'),('after',O/'surfaces/baseline_recolored_final.ply')]:
        mesh=o.io.read_triangle_mesh(str(path));mesh.compute_triangle_normals()
        for i in [26,38,47]:
            im=Image.fromarray(render(mesh,z['poses'][i],z['K']));im.save(O/f'{tag}_surface_{i}.jpg',quality=94)
            im=im.resize((480,360));ImageDraw.Draw(im).text((12,12),f'{tag} | frame {i}',fill='black',stroke_fill='white',stroke_width=2);panels.append(im)
    sheet=Image.new('RGB',(1440,720),'white')
    for k,im in enumerate(panels):sheet.paste(im,((k%3)*480,(k//3)*360))
    sheet.save(O/'surface_color_comparison.jpg',quality=94)
    # Common held-out view: original / previous / selected appearance.
    result=__import__('json').loads((O/'results.json').read_text());selected=O/result['selected_appearance']['name']
    sheet=Image.new('RGB',(2160,540),'white')
    for k,path in enumerate([B/'gs_images/000026.png',B/'gs_colmap_12000/heldout_01.png',selected/'heldout_01.png']):sheet.paste(Image.open(path),(720*k,0))
    sheet.save(O/'appearance_comparison_frame26.jpg',quality=94);print('Six surface renders and appearance comparison saved.')
if __name__=='__main__':main()
