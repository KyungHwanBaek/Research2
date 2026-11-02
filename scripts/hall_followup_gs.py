#!/usr/bin/env python3
"""Matched SfM calibration/control and expanded-context Gaussian experiments."""
from pathlib import Path
import argparse,json,subprocess,time
import cv2,numpy as np,pycolmap
from scipy.spatial.transform import Rotation
O=Path('experiments/student_hall_refinement_20261027');P=Path('experiments/student_hall_refinement_20261026');B=Path('experiments/student_hall_lidar');VAL=[20,26,32,38,44]
def sfm_path(method):
    if method in ['fixed','fixed_noreset']:return P/'stereo_colmap/sparse/0'
    if method in ['focal','principal','principal_noreset']:return O/f'sfm_ba_{method.split("_")[0]}'
    rows=json.loads((O/'sfm_full/summary.json').read_text());row=max(rows,key=lambda r:r['images']);return O/'sfm_full/sparse'/row['component']
def prepare(a):
    rec=pycolmap.Reconstruction(sfm_path(a.method));allims={im.image_id:im for im in rec.images.values()};names={im.name:im for im in allims.values()}
    target={im.image_id for im in allims.values() if 20<=int(Path(im.name).stem)<=45};overlap={i:0 for i in allims}
    for p in rec.points3D.values():
        obs={e.image_id for e in p.track.elements}
        if obs&target:
            for i in obs:overlap[i]+=1
    chosen=[im for im in allims.values() if im.image_id in target or (a.method=='context' and overlap[im.image_id]>=100)]
    val=[names[f'left/{i:06d}.png'] for i in VAL];train=sorted([im for im in chosen if int(Path(im.name).stem) not in VAL],key=lambda im:(int(Path(im.name).stem),im.name))
    total=len(train)+5;stride=int(np.ceil(total/5));it=iter(train);seq=[val[j//stride] if j%stride==0 else next(it) for j in range(total)]
    assert [int(Path(im.name).stem) for j,im in enumerate(seq) if j%stride==0]==VAL
    folder=O/f'input_{a.method}_{a.width}';images=folder/'images';txt=folder/'sparse_text';images.mkdir(parents=True,exist_ok=True);txt.mkdir(exist_ok=True);(folder/'sparse/0').mkdir(parents=True,exist_ok=True)
    cache={};lines=[];manifest=[];h=a.width*3//4
    for j,im in enumerate(seq):
        i=int(Path(im.name).stem);eye=im.name.split('/')[0];source=(O/'sfm_full/images'/im.name) if a.method=='context' else (P/'stereo_colmap/images'/im.name)
        if eye=='left':photo=cv2.imread(str(B/'images'/f'{i:06d}.png'));photo=cv2.resize(photo,(a.width,h),interpolation=cv2.INTER_AREA)
        else:photo=cv2.resize(cv2.imread(str(source)),(a.width,h),interpolation=cv2.INTER_AREA)
        if eye=='left' and a.width==720 and (B/'gs_images'/f'{i:06d}.png').exists():photo=cv2.imread(str(B/'gs_images'/f'{i:06d}.png'))
        cv2.imwrite(str(images/f'{j:06d}.png'),photo)
        if im in train:cache[im.image_id]=photo[:,:,::-1]
        T=im.cam_from_world().matrix();q=Rotation.from_matrix(T[:3,:3]).as_quat();q=np.r_[q[3],q[:3]]
        lines.append(f'{j+1} '+' '.join(map(str,np.r_[q,T[:3,3]]))+f' {im.camera_id} {j:06d}.png\n\n')
        manifest.append(dict(order=j,frame=i,eye=eye,source_image_id=im.image_id,split='validation' if j%stride==0 else 'train',hall_shared_points=overlap[im.image_id]))
    (txt/'images.txt').write_text(''.join(lines));cams=[]
    for cam in rec.cameras.values():cams.append(f'{cam.camera_id} PINHOLE {a.width} {h} '+' '.join(map(str,cam.params*a.width/1440))+'\n')
    (txt/'cameras.txt').write_text(''.join(cams));rows=[];count=0
    for pid,p in rec.points3D.items():
        colors=[]
        for e in p.track.elements:
            if e.image_id not in cache:continue
            im=allims[e.image_id];uv=im.points2D[e.point2D_idx].xy*a.width/1440;ix,iy=np.rint(uv).astype(int)
            if 0<=ix<a.width and 0<=iy<h:colors.append(cache[e.image_id][iy,ix])
        # Same rule for fixed/focal/principal. No validation-image seed colors.
        if len(colors)<2:continue
        color=np.uint8(np.median(colors,axis=0));rows.append(f'{pid} '+' '.join(map(str,p.xyz))+' '+' '.join(map(str,color))+' 0\n');count+=1
    (txt/'points3D.txt').write_text(''.join(rows));subprocess.run(['colmap','model_converter','--input_path',str(txt),'--output_path',str(folder/'sparse/0'),'--output_type','BIN'],check=True)
    data=dict(method=a.method,width=a.width,source_sfm=str(sfm_path(a.method)),images=manifest,train=len(train),validation=5,stride=stride,initial_points=count,seed_colors='Median of training-image observations only; >=2 training observations per seed.',caveat='All source photos used for SfM. Both eyes of five validation timestamps excluded from appearance training and seed coloring. Validation repeatedly used for selection, no independent test set.')
    (folder/'manifest.json').write_text(json.dumps(data,indent=2));print('GS_INPUT',json.dumps({k:v for k,v in data.items() if k!='images'}),flush=True)
def train(a):
    from msplat import Dataset,GaussianTrainer,TrainingConfig,sync,cleanup
    from PIL import Image
    from train_msplat_render_flythrough import interpolate_pose
    data=O/f'input_{a.method}_{a.width}';meta=json.loads((data/'manifest.json').read_text());out=O/f'gs_{a.method}_{a.width}_{a.steps}';out.mkdir(exist_ok=True)
    ds=Dataset(str(data),downscale_factor=1,eval_mode=True,test_every=meta['stride']);assert ds.num_test==5 and ds.num_train==meta['train']
    cfg=TrainingConfig(iterations=a.steps,sh_degree=3,ssim_weight=.2,num_downscales=0,downscale_factor=1,refine_every=100,warmup_length=500,
        reset_alpha_every=1000000 if 'noreset' in a.method else 30,densify_grad_thresh=.0003,densify_size_thresh=.01,stop_screen_size_at=4000,split_screen_size=.05,keep_crs=False,save_every=-1,bg_color=[0,0,0])
    trainer=GaussianTrainer(ds,cfg);history=[];best=-np.inf;start=time.time()
    (out/'manifest.json').write_text(json.dumps(dict(**meta,iterations=a.steps,opacity_reset='disabled' if 'noreset' in a.method else 'every 3000 steps before densification stops',seed=42,evaluation_interval=1000),indent=2))
    def save(im,path):Image.fromarray(np.uint8(np.clip(im,0,1)*255+.5)).save(path)
    def evaluate(step):
        nonlocal best
        sync();m=trainer.evaluate();row=dict(step=step,seconds=time.time()-start,gaussians=trainer.splat_count,**m);history.append(row);(out/'history.json').write_text(json.dumps(history,indent=2));print('GS_EVAL',json.dumps(row),flush=True)
        if m['psnr']>best:
            best=m['psnr'];trainer.save_checkpoint(str(out/'best_checkpoint'));trainer.export_ply(str(out/'best.ply'));trainer.export_splat(str(out/'best.splat'));(out/'best_metrics.json').write_text(json.dumps(row,indent=2))
            for k in range(5):save(trainer.render(k,use_test=True),out/f'heldout_{k:02d}.png')
    def callback(s):print('GS_STEP',s.iteration,s.splat_count,flush=True);evaluate(s.iteration)
    trainer.train(callback,callback_every=1000);sync()
    if not history or history[-1]['step']!=a.steps:evaluate(a.steps)
    trainer.load_checkpoint(str(out/'best_checkpoint'));trainmeta=[v for v in meta['images'] if v['split']=='train'];left=[j for j,v in enumerate(trainmeta) if v['eye']=='left' and 20<=v['frame']<=45];left.sort(key=lambda j:trainmeta[j]['frame'])
    for k in range(72):
        u=k/71*(len(left)-1);j=min(int(u),len(left)-2);pose=interpolate_pose(np.asarray(ds.camera_pose(left[j])),np.asarray(ds.camera_pose(left[j+1])),u-j);save(trainer.render_from_pose(pose,ref_cam_idx=left[0]),out/f'novel_{k:03d}.jpg')
    cleanup();print('GS_DONE',str(out),best,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','train']);p.add_argument('--method',choices=['fixed','focal','principal','fixed_noreset','principal_noreset','context'],required=True);p.add_argument('--width',type=int,default=720);p.add_argument('--steps',type=int,default=12000);a=p.parse_args();prepare(a) if a.stage=='prepare' else train(a)
