#!/usr/bin/env python3
"""Both-eye, higher-resolution COLMAP seed -> native Metal Gaussian rendering."""
from pathlib import Path
import argparse,json,subprocess,time
import numpy as np
import cv2
import pycolmap
from scipy.spatial.transform import Rotation

ROOT=Path('experiments/student_hall_refinement_20261026');SFM=ROOT/'stereo_colmap';OUT=ROOT/'gs3d_stereo'
BASE=Path('experiments/student_hall_lidar');VAL=[20,26,32,38,44]

def prepare():
    models=[pycolmap.Reconstruction(p) for p in (SFM/'sparse').iterdir() if p.is_dir()]
    rec=max(models,key=lambda r:r.num_reg_images());byname={im.name:im for im in rec.images.values()}
    for i in range(20,46):
        if f'left/{i:06d}.png' not in byname:raise RuntimeError(f'Missing reference evaluation/training left frame {i}')
    val=[byname[f'left/{i:06d}.png'] for i in VAL]
    train=[im for im in rec.images.values() if int(Path(im.name).stem) not in VAL]
    train.sort(key=lambda im:(int(Path(im.name).stem),im.name));n=len(val)+len(train);stride=int(np.ceil(n/5));seq=[];it=iter(train)
    for j in range(n):seq.append(val[j//stride] if j%stride==0 else next(it))
    assert [int(Path(im.name).stem) for j,im in enumerate(seq) if j%stride==0]==VAL
    images=OUT/'images';txt=OUT/'sparse_text';images.mkdir(parents=True,exist_ok=True);txt.mkdir(exist_ok=True);(OUT/'sparse/0').mkdir(parents=True,exist_ok=True)
    cameras=[]
    for cam in rec.cameras.values():
        params=cam.params*.5;cameras.append(f'{cam.camera_id} PINHOLE 720 540 '+' '.join(map(str,params))+'\n')
    (txt/'cameras.txt').write_text(''.join(cameras));lines=[];manifest=[]
    for j,im in enumerate(seq):
        i=int(Path(im.name).stem);eye=im.name.split('/')[0]
        if eye=='left':photo=cv2.imread(str(BASE/'gs_images'/f'{i:06d}.png'))
        else:photo=cv2.resize(cv2.imread(str(SFM/'images'/im.name)),(720,540),interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(images/f'{j:06d}.png'),photo);T=im.cam_from_world().matrix();q=Rotation.from_matrix(T[:3,:3]).as_quat();q=np.r_[q[3],q[:3]]
        lines.append(f'{j+1} '+' '.join(map(str,np.r_[q,T[:3,3]]))+f' {im.camera_id} {j:06d}.png\n\n')
        manifest.append(dict(order=j,frame=i,eye=eye,split='validation' if j%stride==0 else 'train'))
    (txt/'images.txt').write_text(''.join(lines))
    (txt/'points3D.txt').write_text(''.join(f'{pid} '+' '.join(map(str,p.xyz))+' '+' '.join(map(str,p.color))+' 0\n' for pid,p in rec.points3D.items()))
    subprocess.run(['colmap','model_converter','--input_path',str(txt),'--output_path',str(OUT/'sparse/0'),'--output_type','BIN'],check=True)
    (OUT/'manifest.json').write_text(json.dumps(dict(images=manifest,train=len(train),validation=5,stride=stride,initial_points=rec.num_points3D(),
        caveat='Both eyes at each validation capture excluded from appearance training. All photos used in SFM. Seed colors from SFM include all source photos. Left validation input files exactly match previous experiment.'),indent=2))
    print('GS3D_INPUT',len(train),'train',5,'validation',rec.num_points3D(),'points',flush=True)

def train(a):
    from msplat import Dataset,GaussianTrainer,TrainingConfig,sync,cleanup
    from PIL import Image
    from train_msplat_render_flythrough import interpolate_pose
    meta=json.loads((OUT/'manifest.json').read_text());folder=ROOT/f'gs3d_stereo_{a.steps}';folder.mkdir(exist_ok=True)
    ds=Dataset(str(OUT),downscale_factor=1.,eval_mode=True,test_every=meta['stride']);assert ds.num_test==5 and ds.num_train==meta['train']
    cfg=TrainingConfig(iterations=a.steps,sh_degree=3,ssim_weight=.2,num_downscales=0,downscale_factor=1.,refine_every=100,
        warmup_length=500,reset_alpha_every=30,densify_grad_thresh=.0003,densify_size_thresh=.01,stop_screen_size_at=4000,split_screen_size=.05,
        keep_crs=False,save_every=-1,bg_color=[0,0,0])
    trainer=GaussianTrainer(ds,cfg);start=time.time();history=[];best=-np.inf
    def save(im,path):Image.fromarray(np.uint8(np.clip(im,0,1)*255+.5)).save(path)
    def evaluate(step):
        nonlocal best
        sync();m=trainer.evaluate();r=dict(step=int(step),elapsed_s=time.time()-start,gaussians=trainer.splat_count,**m);history.append(r)
        (folder/'history.json').write_text(json.dumps(history,indent=2));print('GS3D_EVALUATION',json.dumps(r),flush=True)
        if m['psnr']>best:
            best=m['psnr'];trainer.save_checkpoint(str(folder/'best_checkpoint'));trainer.export_ply(str(folder/'best.ply'))
            trainer.export_splat(str(folder/'best.splat'))
            (folder/'best_metrics.json').write_text(json.dumps(r,indent=2))
            for k in range(5):save(trainer.render(k,use_test=True),folder/f'heldout_{k:02d}.png')
    def callback(s):
        print('step',s.iteration,'gaussians',s.splat_count,'ms',s.ms_per_step,flush=True)
        if s.iteration%2000==0:evaluate(s.iteration)
    trainer.train(callback,callback_every=1000);sync()
    if not history or history[-1]['step']!=a.steps:evaluate(a.steps)
    trainer.load_checkpoint(str(folder/'best_checkpoint'))
    # Save novel views from best selected checkpoint, not merely last step.
    train_meta=[r for r in meta['images'] if r['split']=='train'];left=[j for j,r in enumerate(train_meta) if r['eye']=='left'];left.sort(key=lambda j:train_meta[j]['frame'])
    for k in range(72):
        u=k/71*(len(left)-1);j=min(int(u),len(left)-2)
        P=interpolate_pose(np.asarray(ds.camera_pose(left[j])),np.asarray(ds.camera_pose(left[j+1])),u-j);save(trainer.render_from_pose(P,ref_cam_idx=left[0]),folder/f'novel_{k:03d}.jpg')
    cleanup();print('GS3D_DONE','best',best,'folder',str(folder),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','train']);p.add_argument('--steps',type=int,default=16000);a=p.parse_args();prepare() if a.stage=='prepare' else train(a)
