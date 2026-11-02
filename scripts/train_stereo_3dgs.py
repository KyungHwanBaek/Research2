#!/usr/bin/env python3
"""Train matched sparse/stereo initializations; measure held-out render quality."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw


def save(arr,path):
    Image.fromarray(np.uint8(np.clip(arr,0,1)*255+.5)).save(path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--method',choices=['sparse','stereo','joint'],required=True)
    p.add_argument('--iterations',type=int,default=12000)
    p.add_argument('--resume',type=Path);a=p.parse_args()
    from msplat import Dataset,GaussianTrainer,TrainingConfig,sync,cleanup
    from train_msplat_render_flythrough import interpolate_pose
    out=Path('results/renders')/f'3dgs_0905_{a.method}_{a.iterations}';out.mkdir(parents=True,exist_ok=True)
    scene=Path('exports')/f'3dgs_0905_comparison_{a.method}'
    ds=Dataset(str(scene),downscale_factor=1.,eval_mode=True,test_every=8)
    conf=TrainingConfig(iterations=a.iterations,sh_degree=3,ssim_weight=.2,num_downscales=0,
           downscale_factor=1.,refine_every=100,warmup_length=500,reset_alpha_every=30,
           densify_grad_thresh=.0003,densify_size_thresh=.01,stop_screen_size_at=4000,
           split_screen_size=.05,keep_crs=False,save_every=-1,bg_color=[0.,0.,0.])
    trainer=GaussianTrainer(ds,conf)
    if a.resume:trainer.load_checkpoint(str(a.resume))
    history=[];start=time.time()
    def callback(stat):
        row={'iteration':int(stat.iteration),'gaussians':int(stat.splat_count),'ms_per_step':float(stat.ms_per_step)}
        history.append(row);print(json.dumps(row),flush=True)
    print('loaded',a.method,'train',ds.num_train,'heldout',ds.num_test,flush=True)
    trainer.train(callback,callback_every=500);sync()
    evaluation=trainer.evaluate();print('heldout_evaluation',evaluation,flush=True)
    trainer.export_ply(str(out/'scene.ply'));trainer.save_checkpoint(str(out/'checkpoint'))
    for k in range(ds.num_test):save(trainer.render(k,use_test=True),out/f'heldout_{k:03d}.png')
    for k in [0,ds.num_train//2,ds.num_train-1]:save(trainer.render(k),out/f'train_{k:03d}.png')
    # Smooth novel camera path through the trained local segment.
    frames=[]
    for k in range(72):
        u=k/71*(ds.num_train-1);j=min(int(u),ds.num_train-2)
        P=interpolate_pose(np.asarray(ds.camera_pose(j)),np.asarray(ds.camera_pose(j+1)),u-j)
        img=trainer.render_from_pose(P,ref_cam_idx=0)
        path=out/f'novel_{k:03d}.jpg';save(img,path)
        im=Image.open(path);im.thumbnail((540,405));frames.append(im.copy())
    frames[0].save(out/'flythrough.gif',save_all=True,append_images=frames[1:],duration=85,loop=0)
    report={'method':a.method,'iterations':a.iterations,'resolution':[720,540],'train_views':ds.num_train,
            'heldout_views':ds.num_test,'heldout_metrics':evaluation,'final_gaussians':trainer.splat_count,
            'elapsed_s':time.time()-start,'history':history,
            'note':'Identical images/poses/split/config for both methods; only point initialization differs. Held-out RGB views excluded from training; stereo initialization also excludes their depth. Poses were estimated using all source frames.'}
    (out/'metrics.json').write_text(json.dumps(report,indent=2))
    imgs=sorted(out.glob('heldout_*.png'));thumbs=[]
    for path in imgs:
        im=Image.open(path);im.thumbnail((360,270));thumbs.append(im)
    sheet=Image.new('RGB',(4*360,((len(thumbs)+3)//4)*270))
    for k,im in enumerate(thumbs):sheet.paste(im,((k%4)*360,(k//4)*270))
    sheet.save(out/'heldout_contact_sheet.jpg')
    cleanup();print('completed',out,flush=True)


if __name__=='__main__':main()
