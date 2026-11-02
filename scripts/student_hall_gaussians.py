#!/usr/bin/env python3
"""Matched local 3DGS trials: COLMAP landmarks vs raw-LiDAR initialization.

Appearance-only training; no claim of LiDAR depth-loss implementation.
"""
from pathlib import Path
import argparse,json,subprocess,time
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from reconstruct_student_hall_lidar import OUT,data,transform,pixels_xyz,cloud

def prepare():
    ps=np.load(OUT/'joint_poses.npz');poses=ps['poses'];K=np.load(OUT/'calibration.npz')['K'];indices=list(range(20,46))
    images=OUT/'gs_images';images.mkdir(exist_ok=True)
    test=[i for j,i in enumerate(indices) if j%6==0];train=[i for i in indices if i not in test]
    for i in indices:
        im=cv2.imread(str(OUT/'images'/f'{i:06d}.png'));cv2.imwrite(str(images/f'{i:06d}.png'),cv2.resize(im,(720,540),interpolation=cv2.INTER_AREA))
    xs=[];cs=[]
    for i in train:
        depth=data(i)['depth'];yy,xx=np.mgrid[0:720:4,0:960:4];uv=np.column_stack((xx.ravel(),yy.ravel()))
        x=pixels_xyz(depth,uv,K);ok=(x[:,2]>.8)&(x[:,2]<35);x=x[ok];uv=uv[ok]
        xs.append(transform(x,poses[i]));cs.append(cv2.imread(str(OUT/'images'/f'{i:06d}.png'))[uv[:,1],uv[:,0],::-1])
    import open3d as o
    dense=o.geometry.PointCloud(o.utility.Vector3dVector(np.concatenate(xs)));dense.colors=o.utility.Vector3dVector(np.concatenate(cs)/255)
    dense=dense.voxel_down_sample(.07);x=np.asarray(dense.points);c=np.uint8(np.asarray(dense.colors)*255)
    rng=np.random.default_rng(42);ids=rng.choice(len(x),min(50000,len(x)),replace=False);dx,dc=x[ids],c[ids]
    x=ps['points'];c=np.full((len(x),3),128,np.uint8);seen=np.zeros(len(x),int)
    for i in train:
        cam=transform(x,np.linalg.inv(poses[i]));uv=cam[:,:2]/np.maximum(cam[:,2:],.001)*[K[0,0],K[1,1]]+K[:2,2];pix=np.rint(uv).astype(int)
        ok=(cam[:,2]>.7)&(cam[:,2]<60)&(pix[:,0]>=0)&(pix[:,0]<960)&(pix[:,1]>=0)&(pix[:,1]<720)
        ids=np.flatnonzero(ok);im=cv2.imread(str(OUT/'images'/f'{i:06d}.png'));c[ids]=im[pix[ids,1],pix[ids,0],::-1];seen[ids]+=1
    sx,sc=x[seen>=3],c[seen>=3]
    import pycolmap
    native=pycolmap.Reconstruction('experiments/rgb_colmap_well_lit_outdoor_768_exhaustive_fixed/sparse/0')
    metric=np.load(OUT/'colmap_component0.npz');anchor=np.linalg.inv(metric['poses'][0]);native_poses={int(i):anchor@p for i,p in zip(metric['indices'],metric['poses'])}
    nx=transform(np.array([p.xyz for p in native.points3D.values()])*metric['scale'],anchor)
    nc=np.array([p.color for p in native.points3D.values()])
    for method,xyz,rgb in [('sparse',sx,sc),('lidar',dx,dc),('colmap',nx,nc)]:
        root=OUT/f'gs_{method}';text=root/'sparse_text';text.mkdir(parents=True,exist_ok=True);(root/'sparse/0').mkdir(parents=True,exist_ok=True)
        if not (root/'images').exists():(root/'images').symlink_to(images.resolve(),target_is_directory=True)
        k=K.copy();k[:2]*=.75
        (text/'cameras.txt').write_text('1 PINHOLE 720 540 '+' '.join(map(str,[k[0,0],k[1,1],k[0,2],k[1,2]]))+'\n')
        lines=[]
        for j,i in enumerate(indices):
            T=np.linalg.inv(native_poses[i] if method=='colmap' else poses[i]);q=Rotation.from_matrix(T[:3,:3]).as_quat();q=np.r_[q[3],q[:3]]
            lines.append(f'{j+1} '+' '.join(map(str,np.r_[q,T[:3,3]]))+f' 1 {i:06d}.png\n\n')
        (text/'images.txt').write_text(''.join(lines))
        (text/'points3D.txt').write_text(''.join(f'{j+1} '+' '.join(map(str,p))+' '+' '.join(map(str,c))+' 0\n' for j,(p,c) in enumerate(zip(xyz,rgb))))
        subprocess.run(['colmap','model_converter','--input_path',str(text),'--output_path',str(root/'sparse/0'),'--output_type','BIN'],check=True)
        (root/'manifest.json').write_text(json.dumps({'images':indices,'train':train,'test':test,'initial_points':len(xyz),
            'camera_poses':('original COLMAP, metric scaled' if method=='colmap' else 'joint RGB/COLMAP + LiDAR plane refinement'),'method':method,
            'note':'Only initialization differs. Initial colors/depth use training views. Camera estimation and original SfM used all source images.'},indent=2))

def train(method,iterations):
    from msplat import Dataset,GaussianTrainer,TrainingConfig,sync,cleanup
    from train_msplat_render_flythrough import interpolate_pose
    from PIL import Image
    root=OUT/f'gs_{method}';out=OUT/f'gs_{method}_{iterations}';out.mkdir(exist_ok=True)
    ds=Dataset(str(root),downscale_factor=1.,eval_mode=True,test_every=6)
    conf=TrainingConfig(iterations=iterations,sh_degree=3,ssim_weight=.2,num_downscales=0,
        downscale_factor=1.,refine_every=100,warmup_length=500,reset_alpha_every=30,densify_grad_thresh=.0003,
        densify_size_thresh=.01,stop_screen_size_at=4000,split_screen_size=.05,keep_crs=False,save_every=-1,bg_color=[0,0,0])
    trainer=GaussianTrainer(ds,conf);start=time.time()
    previous=OUT/f'gs_{method}_12000/checkpoint'
    if iterations>12000 and previous.exists():trainer.load_checkpoint(str(previous))
    print('train_start',method,'train',ds.num_train,'test',ds.num_test,flush=True)
    trainer.train(lambda s: print('step',s.iteration,s.splat_count,round(s.ms_per_step,1),flush=True),callback_every=1000);sync()
    metrics=trainer.evaluate();trainer.export_ply(str(out/'scene.ply'));trainer.save_checkpoint(str(out/'checkpoint'))
    def save(im,path):Image.fromarray(np.uint8(np.clip(im,0,1)*255+.5)).save(path)
    for k in range(ds.num_test):save(trainer.render(k,use_test=True),out/f'heldout_{k:02d}.png')
    for k in range(72):
        u=k/71*(ds.num_train-1);j=min(int(u),ds.num_train-2);P=interpolate_pose(np.asarray(ds.camera_pose(j)),np.asarray(ds.camera_pose(j+1)),u-j)
        save(trainer.render_from_pose(P,ref_cam_idx=0),out/f'novel_{k:03d}.jpg')
    report={'method':method,'iterations':iterations,'metrics':metrics,'train_views':ds.num_train,'test_views':ds.num_test,
        'final_gaussians':trainer.splat_count,'elapsed_s':time.time()-start,'note':'RGB loss only; LiDAR is initialization / pose support, not a depth training loss. Poses estimated with all source images.'}
    (out/'metrics.json').write_text(json.dumps(report,indent=2));print('evaluation',report,flush=True);cleanup()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','sparse','lidar','colmap']);p.add_argument('--iterations',type=int,default=12000);a=p.parse_args()
    cv2.setNumThreads(4)
    if a.stage=='prepare':prepare()
    else:train(a.stage,a.iterations)
