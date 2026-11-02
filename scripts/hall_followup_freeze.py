#!/usr/bin/env python3
"""Refine the best early checkpoint without further opacity resets/densifying."""
from pathlib import Path
import argparse,json,time
import numpy as np,cv2
from skimage.metrics import structural_similarity
from hall_followup_gs import prepare,O,B,VAL

def main(a):
    from msplat import Dataset,GaussianTrainer,TrainingConfig,sync,cleanup
    from train_msplat_render_flythrough import interpolate_pose
    from PIL import Image
    parent=O/f'gs_{a.method}_720_12000';best_parent=json.loads((parent/'best_metrics.json').read_text());initial_step=best_parent['step'];end=initial_step+a.steps
    data=O/f'input_{a.method}_{a.width}'
    if not (data/'manifest.json').exists():prepare(a)
    meta=json.loads((data/'manifest.json').read_text());out=O/f'freeze_{a.method}_{a.width}';out.mkdir(exist_ok=True)
    ds=Dataset(str(data),downscale_factor=1,eval_mode=True,test_every=meta['stride']);cfg=TrainingConfig(iterations=end,sh_degree=3,ssim_weight=.2,num_downscales=0,downscale_factor=1.,refine_every=100,warmup_length=100000000,reset_alpha_every=30,densify_grad_thresh=.0003,densify_size_thresh=.01,stop_screen_size_at=4000,split_screen_size=.05,keep_crs=False,save_every=-1,bg_color=[0,0,0])
    trainer=GaussianTrainer(ds,cfg);trainer.load_checkpoint(str(parent/'best_checkpoint'));start=time.time();best=-np.inf;history=[]
    (out/'manifest.json').write_text(json.dumps(dict(method=a.method,width=a.width,parent=str(parent),parent_metrics=best_parent,additional_steps=a.steps,description='Best early checkpoint retained; no further opacity resets, split/clone/prune. Other Gaussian parameters continue optimizing. Same data/split. Downsample to 720x540 for comparison.'),indent=2))
    def save(im,path):Image.fromarray(np.uint8(np.clip(im,0,1)*255+.5)).save(path)
    def evaluate(step):
        nonlocal best
        sync();rows=[];renders=[]
        for k,i in enumerate(VAL):
            im=np.asarray(trainer.render(k,use_test=True));im=cv2.resize(im,(720,540),interpolation=cv2.INTER_AREA);target=cv2.imread(str(B/'gs_images'/f'{i:06d}.png'))[:,:,::-1].astype(np.float32)/255
            rows.append(dict(frame=i,psnr=float(-10*np.log10(np.mean((im-target)**2))),ssim=float(structural_similarity(target,im,data_range=1,channel_axis=2,gaussian_weights=True,sigma=1.5,use_sample_covariance=False))));renders.append(im)
        row=dict(step=step,psnr=float(np.mean([r['psnr'] for r in rows])),ssim=float(np.mean([r['ssim'] for r in rows])),views=rows,gaussians=trainer.splat_count,seconds=time.time()-start);history.append(row);(out/'history.json').write_text(json.dumps(history,indent=2));print('FREEZE_EVAL',json.dumps(row),flush=True)
        if row['psnr']>best:
            best=row['psnr'];trainer.save_checkpoint(str(out/'best_checkpoint'));trainer.export_ply(str(out/'best.ply'));trainer.export_splat(str(out/'best.splat'));(out/'best_metrics.json').write_text(json.dumps(row,indent=2))
            for k,im in enumerate(renders):save(im,out/f'heldout_{k:02d}.png')
    evaluate(initial_step)
    for j in range(a.steps):
        stats=trainer.step()
        if (j+1)%1000==0:evaluate(stats.iteration)
    trainer.load_checkpoint(str(out/'best_checkpoint'));trainmeta=[r for r in meta['images'] if r['split']=='train'];left=[j for j,v in enumerate(trainmeta) if v['eye']=='left' and 20<=v['frame']<=45];left.sort(key=lambda j:trainmeta[j]['frame'])
    for k in range(72):
        u=k/71*(len(left)-1);j=min(int(u),len(left)-2);pose=interpolate_pose(np.asarray(ds.camera_pose(left[j])),np.asarray(ds.camera_pose(left[j+1])),u-j);save(trainer.render_from_pose(pose,ref_cam_idx=left[0]),out/f'novel_{k:03d}.jpg')
    cleanup();print('FREEZE_DONE',str(out),best,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--method',required=True,choices=['fixed','focal','principal','context']);p.add_argument('--width',type=int,default=960);p.add_argument('--steps',type=int,default=6000);main(p.parse_args())
