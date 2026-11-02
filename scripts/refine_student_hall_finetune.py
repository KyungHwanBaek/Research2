#!/usr/bin/env python3
"""Freeze Gaussian count and refine appearance at 720 or 960px; same 5 views.

Rendering/validation comparisons are all downsampled to the original 720px.
"""
from pathlib import Path
import argparse,json,time,subprocess
import numpy as np
import cv2
from skimage.metrics import structural_similarity

R=Path('experiments/student_hall_refinement_20261026');B=Path('experiments/student_hall_lidar');SOURCE=R/'gs3d_stereo'
def prepare(width):
    meta=json.loads((SOURCE/'manifest.json').read_text());out=R/f'gs3d_input_{width}';(out/'images').mkdir(parents=True,exist_ok=True);txt=out/'sparse_text';txt.mkdir(exist_ok=True);(out/'sparse/0').mkdir(parents=True,exist_ok=True)
    for row in meta['images']:
        i=row['frame'];eye=row['eye'];j=row['order']
        if eye=='left':photo=cv2.imread(str(B/'images'/f'{i:06d}.png'))
        else:photo=cv2.imread(str(R/'stereo_colmap/images/right'/f'{i:06d}.png'))
        photo=cv2.resize(photo,(width,width*3//4),interpolation=cv2.INTER_AREA);cv2.imwrite(str(out/'images'/f'{j:06d}.png'),photo)
    for name in ['images.txt','points3D.txt']:(txt/name).write_bytes((SOURCE/'sparse_text'/name).read_bytes())
    lines=[]
    for line in (SOURCE/'sparse_text/cameras.txt').read_text().splitlines():
        a=line.split();params=np.array(a[4:],float)*width/720;lines.append(f'{a[0]} PINHOLE {width} {width*3//4} '+' '.join(map(str,params))+'\n')
    (txt/'cameras.txt').write_text(''.join(lines));subprocess.run(['colmap','model_converter','--input_path',str(txt),'--output_path',str(out/'sparse/0'),'--output_type','BIN'],check=True)
    return out,meta

def main(a):
    from msplat import Dataset,GaussianTrainer,TrainingConfig,sync,cleanup
    from train_msplat_render_flythrough import interpolate_pose
    from PIL import Image
    dataset,meta=prepare(a.width);parent=R/'gs3d_stereo_16000';old=json.loads((parent/'best_metrics.json').read_text());start_step=old['step'];end=start_step+a.steps
    out=R/f'gs3d_fine_{a.width}';out.mkdir(exist_ok=True);ds=Dataset(str(dataset),downscale_factor=1,eval_mode=True,test_every=meta['stride'])
    cfg=TrainingConfig(iterations=end,sh_degree=3,ssim_weight=.2,num_downscales=0,downscale_factor=1.,refine_every=100,warmup_length=100000000,
        reset_alpha_every=30,densify_grad_thresh=.0003,densify_size_thresh=.01,stop_screen_size_at=4000,split_screen_size=.05,keep_crs=False,save_every=-1,bg_color=[0,0,0])
    trainer=GaussianTrainer(ds,cfg);trainer.load_checkpoint(str(parent/'best_checkpoint'));start=time.time();history=[];best=-np.inf
    (out/'manifest.json').write_text(json.dumps(dict(width=a.width,additional_steps=a.steps,start_step=start_step,end_step=end,source_checkpoint='gs3d_stereo_16000/best_checkpoint',
        source_metrics=old,freeze='No split/clone/prune/opacity reset; Gaussian count fixed, means/scales/quaternions/SH/opacity still optimized.',
        validation='Same five left photos, both eyes excluded from training at these timestamps. Render downsampled to 720x540 using area filter.'),indent=2))
    def save(im,path):Image.fromarray(np.uint8(np.clip(im,0,1)*255+.5)).save(path)
    def evaluate(step):
        nonlocal best
        sync();rows=[];renders=[]
        for k,i in enumerate([20,26,32,38,44]):
            pred=np.asarray(trainer.render(k,use_test=True));pred=cv2.resize(pred,(720,540),interpolation=cv2.INTER_AREA)
            truth=cv2.imread(str(B/'gs_images'/f'{i:06d}.png'))[:,:,::-1].astype(np.float32)/255
            p=float(-10*np.log10(np.mean((pred-truth)**2)));s=float(structural_similarity(truth,pred,data_range=1,channel_axis=2,gaussian_weights=True,sigma=1.5,use_sample_covariance=False))
            rows.append(dict(frame=i,psnr=p,ssim=s));renders.append(pred)
        row=dict(step=step,width=a.width,psnr=float(np.mean([v['psnr'] for v in rows])),ssim=float(np.mean([v['ssim'] for v in rows])),views=rows,gaussians=trainer.splat_count,elapsed_s=time.time()-start)
        history.append(row);(out/'history.json').write_text(json.dumps(history,indent=2));print('FINE_EVALUATION',json.dumps(row),flush=True)
        if row['psnr']>best:
            best=row['psnr'];trainer.save_checkpoint(str(out/'best_checkpoint'));trainer.export_ply(str(out/'best.ply'));trainer.export_splat(str(out/'best.splat'))
            (out/'best_metrics.json').write_text(json.dumps(row,indent=2))
            for k,im in enumerate(renders):save(im,out/f'heldout_{k:02d}.png')
    evaluate(start_step)
    def callback(s):
        print('fine_step',s.iteration,s.splat_count,flush=True)
        if (s.iteration-start_step)%2000==0:evaluate(s.iteration)
    # step() permits an exact additional-step budget regardless of checkpoint
    # metadata. No changes to parent checkpoints or baseline experiments.
    for j in range(a.steps):
        stats=trainer.step()
        if (j+1)%1000==0:callback(stats)
    if history[-1]['step']!=end:evaluate(end)
    trainer.load_checkpoint(str(out/'best_checkpoint'));train=[v for v in meta['images'] if v['split']=='train'];left=[j for j,v in enumerate(train) if v['eye']=='left'];left.sort(key=lambda j:train[j]['frame'])
    for k in range(72):
        u=k/71*(len(left)-1);j=min(int(u),len(left)-2);P=interpolate_pose(np.asarray(ds.camera_pose(left[j])),np.asarray(ds.camera_pose(left[j+1])),u-j)
        save(trainer.render_from_pose(P,ref_cam_idx=left[0]),out/f'novel_{k:03d}.jpg')
    cleanup();print('FINE_DONE',str(out),best,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--width',type=int,choices=[720,960],required=True);p.add_argument('--steps',type=int,default=6000);main(p.parse_args())
