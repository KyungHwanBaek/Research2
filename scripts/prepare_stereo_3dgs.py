#!/usr/bin/env python3
"""Package identical camera/image splits with sparse and stereo initializations."""
import json
from pathlib import Path
import subprocess
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from improve_stereo_reconstruction import Stereo
from fuse_stereo_reconstruction import read_model,voxel_average


def main():
    cv2.setNumThreads(4)
    s=Stereo('data/extracted/sampled_540p/09-05-17-34-36','experiments/stereo_improved_0905/depth')
    model,points,width=read_model(Path('experiments/rgb_colmap_0905_all_s1_360_seq20_fixed/sparse/2_text_for_summary'))
    selected=[i for i in sorted(model) if 140<=i<=440 and i%5==0]
    ratios=[]
    for i in selected[::4]:
        obs=model[i]['obs'];obs=obs[obs[:,2]>=0]
        xyz=np.array([points[int(p)] for p in obs[:,2]])
        cam=(xyz@model[i]['w2c'][:3,:3].T+model[i]['w2c'][:3,3])@s.R1.T
        uv=obs[:,:2]*2
        rect=cv2.undistortPoints(uv[:,None],s.K,None,R=s.R1,P=s.P1)[:,0]
        xy=np.rint(rect).astype(int)
        inside=(xy[:,0]>=0)&(xy[:,0]<s.size[0])&(xy[:,1]>=0)&(xy[:,1]<s.size[1])&(cam[:,2]>0)
        dd=s.combined_depth(i)[xy[inside,1],xy[inside,0]];cc=cam[inside,2]
        ratios.extend((dd[dd>0]/cc[dd>0]).tolist())
    scale=float(np.median(ratios)); print('local_metric_scale',scale,flush=True)
    D=np.eye(4);D[:3,:3]=s.R1
    poses=[]
    for i in selected:
        T=model[i]['w2c'].copy();T[:3,3]*=scale;poses.append(np.linalg.inv(D@T))
    poses=np.array(poses)
    xyzs=[];rgbs=[]
    for j,i in enumerate(selected):
        # Exclude either possible evaluation indexing convention from depth initialization.
        if j%8==0 or (j+1)%8==0:continue
        dep=s.combined_depth(i)
        yy,xx=np.mgrid[0:dep.shape[0]:3,0:dep.shape[1]:3]
        uv=np.column_stack((xx.ravel(),yy.ravel())).astype(np.float32)
        xyz=s.points(dep,uv);valid=(xyz[:,2]>0)&(xyz[:,2]<25)
        uv,xyz=uv[valid],xyz[valid]
        world=xyz@poses[j,:3,:3].T+poses[j,:3,3]
        support=np.zeros(len(world),bool)
        for n in [j-1,j+1]:
            if n<0 or n>=len(selected) or n%8==0 or (n+1)%8==0:continue
            cam=(world-poses[n,:3,3])@poses[n,:3,:3]
            proj=cam[:,:2]/np.maximum(cam[:,2:],1e-6)*np.array([s.rectK[0,0],s.rectK[1,1]])+s.rectK[:2,2]
            xy=np.rint(proj).astype(int)
            mask=(cam[:,2]>.5)&(xy[:,0]>=0)&(xy[:,0]<s.size[0])&(xy[:,1]>=0)&(xy[:,1]<s.size[1])
            ids=np.flatnonzero(mask);dn=s.combined_depth(selected[n])[xy[ids,1],xy[ids,0]]
            support[ids]=(dn>0)&(np.abs(dn-cam[ids,2])<.045*cam[ids,2])|support[ids]
        col=s.image(i)[uv[:,1].astype(int),uv[:,0].astype(int),::-1]
        xyzs.append(world[support]);rgbs.append(col[support])
    dense,col,_=voxel_average(np.concatenate(xyzs),np.concatenate(rgbs),.07)
    rng=np.random.default_rng(42);ix=rng.choice(len(dense),min(45000,len(dense)),replace=False)
    dense,col=dense[ix],col[ix]
    # Keep sparse colors directly from the reference reconstruction.
    sparse=[];sparse_colors=[]
    pointsfile=Path('experiments/rgb_colmap_0905_all_s1_360_seq20_fixed/sparse/2_text_for_summary/points3D.txt')
    for line in pointsfile.read_text().splitlines():
        if not line or line.startswith('#'):continue
        p=line.split();sparse.append(np.array(p[1:4],float)*scale);sparse_colors.append(np.array(p[4:7],int))
    sparse=np.array(sparse);sparse_colors=np.array(sparse_colors)
    used=[p for p in sorted(model) if p in selected]
    shared=Path('exports/3dgs_stereo_0905_images');shared.mkdir(exist_ok=True)
    for i in selected:cv2.imwrite(str(shared/f'{i:06d}.png'),s.image(i))
    for name,xyz,rgb in [('sparse',sparse,sparse_colors),('stereo',dense,col)]:
        out=Path('exports')/f'3dgs_0905_comparison_{name}'
        (out/'sparse/0').mkdir(parents=True,exist_ok=True)
        if not (out/'images').exists():(out/'images').symlink_to(shared.resolve(),target_is_directory=True)
        text=out/'sparse_text';text.mkdir(exist_ok=True)
        (text/'cameras.txt').write_text(f'1 PINHOLE {s.size[0]} {s.size[1]} {s.rectK[0,0]} {s.rectK[1,1]} {s.rectK[0,2]} {s.rectK[1,2]}\n')
        lines=[]
        for j,(i,P) in enumerate(zip(selected,poses)):
            T=np.linalg.inv(P);q=Rotation.from_matrix(T[:3,:3]).as_quat();q=np.r_[q[3],q[:3]]
            lines.append(f'{j+1} '+ ' '.join(str(x) for x in np.r_[q,T[:3,3]])+f' 1 {i:06d}.png\n\n')
        (text/'images.txt').write_text(''.join(lines))
        (text/'points3D.txt').write_text(''.join(f'{k+1} '+ ' '.join(str(x) for x in p)+' '+' '.join(str(int(v)) for v in c)+' 0\n' for k,(p,c) in enumerate(zip(xyz,rgb))))
        subprocess.run(['colmap','model_converter','--input_path',str(text),'--output_path',str(out/'sparse/0'),'--output_type','BIN'],check=True)
        (out/'manifest.json').write_text(json.dumps({'images':selected,'initial_points':len(xyz),'method':name,'scale':scale,
                'evaluation':'Dataset(eval_mode=True,test_every=8). Stereo depth initialization excludes indices j%8==0 and (j+1)%8==0.',
                'poses':'Original locally bundle-adjusted COLMAP poses, rectified and scaled using stereo. No LiDAR.'},indent=2))
        print('exported',out,'images',len(selected),'points',len(xyz),flush=True)


if __name__=='__main__':main()
