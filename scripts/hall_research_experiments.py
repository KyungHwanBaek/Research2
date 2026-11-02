#!/usr/bin/env python3
"""Controlled native-resolution, tile-capacity, and stereo-color experiments."""
from pathlib import Path
import argparse,hashlib,json,shutil,sys,time
import cv2,numpy as np
from PIL import Image
from skimage.metrics import structural_similarity
O=Path('experiments/student_hall_research_20261027'); P=Path('experiments/student_hall_refinement_20261027')
B=Path('experiments/student_hall_lidar'); PRE=Path('experiments/student_hall_refinement_20261026');VAL=[20,26,32,38,44]
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def read(p):return json.loads(p.read_text())
def metric(x,y):
    x=x.astype(np.float64);y=y.astype(np.float64)
    return dict(psnr=float(-10*np.log10(np.mean((x-y)**2))),ssim=float(structural_similarity(y,x,data_range=1,channel_axis=2,gaussian_weights=True,sigma=1.5,use_sample_covariance=False)))
def prepare():
    import pycolmap
    from scipy.optimize import least_squares
    cv2.setNumThreads(4);meta=read(P/'input_principal_960/manifest.json');sfm=pycolmap.Reconstruction(P/'sfm_ba_principal');imbyid=dict(sfm.images.items())
    trainids={r['source_image_id'] for r in meta['images'] if r['split']=='train'}
    cache={i:cv2.imread(str(PRE/'stereo_colmap/images'/imbyid[i].name))[:,:,::-1].astype(np.float32)/255 for i in trainids}
    samples={i:[] for i in trainids if imbyid[i].name.startswith('right/')}
    for point in sfm.points3D.values():
        obs=[]
        for e in point.track.elements:
            if e.image_id not in cache:continue
            xy=np.rint(imbyid[e.image_id].points2D[e.point2D_idx].xy).astype(int);x,y=xy
            if x<2 or x>=1438 or y<2 or y>=1078:continue
            patch=cache[e.image_id][y-1:y+2,x-1:x+2];col=np.median(patch,axis=(0,1));obs.append((e.image_id,col))
        left=[c for i,c in obs if imbyid[i].name.startswith('left/')]
        if len(left)<2:continue
        ref=np.median(left,axis=0)
        for i,c in obs:
            if i in samples and np.all((c>.03)&(c<.97)&(ref>.03)&(ref<.97)):samples[i].append((c,ref))
    fits={};fitrows=[]
    for i,rows in sorted(samples.items()):
        xy=np.asarray(rows); gains=[];biases=[]
        if len(xy)<100:gains=[1.]*3;biases=[0.]*3
        else:
            for c in range(3):
                x,y=xy[:,0,c],xy[:,1,c]
                def residual(v):return np.r_[v[0]*x+v[1]-y,.25*(v[0]-1),.5*v[1]]
                fit=least_squares(residual,[1.,0.],bounds=([.7,-.1],[1.4,.1]),loss='soft_l1',f_scale=.03)
                gains.append(float(fit.x[0]));biases.append(float(fit.x[1]))
        fits[i]=(np.array(gains),np.array(biases));before=float(np.median(np.abs(xy[:,0]-xy[:,1]))) if len(xy) else None
        after=float(np.median(np.abs(xy[:,0]*gains+biases-xy[:,1]))) if len(xy) else None
        fitrows.append(dict(source_image_id=i,name=imbyid[i].name,training_tracks=len(rows),gain=gains,bias=biases,median_patch_abs_error_before=before,median_patch_abs_error_after=after))
    dump(O/'stereo_color_fits.json',dict(method='Robust per-channel affine fit of right training images to medians of >=2 left training observations. No validation observations. Patch=3x3; soft-L1 0.03; regularized identity. This is preprocessing, not the official learned per-image exposure algorithm.',fits=fitrows))
    for width,color in [(960,False),(1440,False),(1440,True)]:
        name=f'native_{width}'+('_color' if color else '');out=O/name;shutil.copytree(P/'input_principal_960/sparse',out/'sparse');(out/'images').mkdir()
        rec=pycolmap.Reconstruction(out/'sparse/0')
        for cam in rec.cameras.values():cam.params*=width/960;cam.width=width;cam.height=width*3//4
        rec.write(out/'sparse/0')
        for row in meta['images']:
            side=row['eye'];frame=row['frame'];source=PRE/'stereo_colmap/images'/side/f'{frame:06d}.png';im=cv2.imread(str(source));assert im.shape==(1080,1440,3)
            if color and side=='right' and row['split']=='train':
                g,b=fits[row['source_image_id']];rgb=im[:,:,::-1].astype(np.float64)/255;im=np.uint8(np.clip(rgb*g+b,0,1)*255+.5)[:,:,::-1]
            if width!=1440:im=cv2.resize(im,(width,width*3//4),interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out/'images'/f'{row["order"]:06d}.png'),im)
        dump(out/'manifest.json',dict(**{k:v for k,v in meta.items() if k!='width'},width=width,native_source='Original undistorted 1440x1080 left AND right photos, not an upscaled 960px cache.',color_normalized=color))
        print('INPUT_PREPARED',name,flush=True)
def train(a):
    if a.engine=='wide':sys.path.insert(0,str((O/'native_package').resolve()))
    import msplat
    from msplat import Dataset,GaussianTrainer,TrainingConfig,sync,cleanup
    from train_msplat_render_flythrough import interpolate_pose
    data=P/'input_principal_960' if a.input=='legacy_960' else O/a.input;meta=read(data/'manifest.json');parent=P/'freeze_principal_960';parent_step=read(parent/'best_metrics.json')['step']
    out=O/a.name;out.mkdir();ds=Dataset(str(data),downscale_factor=1,eval_mode=True,test_every=meta['stride']);assert ds.num_train==42 and ds.num_test==5
    cfg=TrainingConfig(iterations=18000 if a.densify else parent_step+max(a.steps,4000),sh_degree=3,ssim_weight=.2,num_downscales=0,downscale_factor=1.,warmup_length=0 if a.densify else 100000000,refine_every=500,reset_alpha_every=1000000,densify_grad_thresh=.0003,densify_size_thresh=.01,stop_screen_size_at=0,save_every=-1,bg_color=[0,0,0])
    trainer=GaussianTrainer(ds,cfg);trainer.load_checkpoint(str(parent/'best_checkpoint'))
    if a.lr!=1:assert a.engine=='wide';trainer.scale_learning_rates(a.lr)
    manifest=dict(input=str(data),engine=a.engine,engine_module=msplat.__file__,tile_capacity=4096 if a.engine=='wide' else 2048,lr_factor=a.lr,parent=str(parent),parent_step=parent_step,additional_steps=a.steps,densification=a.densify,description='Continue Gaussian attributes; no validation-pose or exposure fitting. Validation reused for model selection. Optional progressive standard densification every500, threshold0.0003, stop9000; opacity resets disabled. Not AbsGS.',checkpoint_schedule_max_step=cfg.iterations)
    dump(out/'manifest.json',manifest);history=[];best=-np.inf;start=time.time()
    def save(im,p):Image.fromarray(np.uint8(np.clip(im,0,1)*255+.5)).save(p)
    def evaluate(step):
        nonlocal best
        sync();rows=[];renders=[];over=[]
        for k,frame in enumerate(VAL):
            im=np.asarray(trainer.render(k,use_test=True));assert np.isfinite(im).all();im=cv2.resize(im,(720,540),interpolation=cv2.INTER_AREA);im=np.uint8(np.clip(im,0,1)*255+.5).astype(np.float64)/255
            target=np.asarray(Image.open(B/'gs_images'/f'{frame:06d}.png')).astype(np.float64)/255;rows.append(dict(frame=frame,**metric(im,target)));renders.append(im)
            if a.engine=='wide':over.append(int(msplat._core.tile_overflow_status()))
        row=dict(step=step,psnr=float(np.mean([r['psnr'] for r in rows])),ssim=float(np.mean([r['ssim'] for r in rows])),views=rows,gaussians=trainer.splat_count,seconds=time.time()-start,validation_overflow_flags=over);history.append(row);dump(out/'history.json',history);print('RESEARCH_EVAL',json.dumps(row),flush=True)
        if row['psnr']>best:
            best=row['psnr'];trainer.save_checkpoint(str(out/'best_checkpoint'));trainer.export_ply(str(out/'best.ply'));trainer.export_splat(str(out/'best.splat'));dump(out/'best_metrics.json',row)
            for k,im in enumerate(renders):save(im,out/f'heldout_{k:02d}.png')
    evaluate(parent_step)
    for j in range(a.steps):
        stat=trainer.step()
        if (j+1)%a.interval==0 or j+1==a.steps:evaluate(stat.iteration)
        if stat.splat_count>600000:
            print('STOPPING_AT_POINT_BUDGET',stat.splat_count,flush=True);break
    trainer.load_checkpoint(str(out/'best_checkpoint'))
    trainmeta=[r for r in meta['images'] if r['split']=='train'];left=[j for j,v in enumerate(trainmeta) if v['eye']=='left'];left.sort(key=lambda j:trainmeta[j]['frame'])
    for k in range(72):
        u=k/71*(len(left)-1);j=min(int(u),len(left)-2);pose=interpolate_pose(np.asarray(ds.camera_pose(left[j])),np.asarray(ds.camera_pose(left[j+1])),u-j);im=np.asarray(trainer.render_from_pose(pose,ref_cam_idx=left[0]));im=cv2.resize(im,(960,720),interpolation=cv2.INTER_AREA);save(im,out/f'novel_{k:03d}.jpg')
    cleanup();print('RESEARCH_DONE',a.name,best,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','train']);p.add_argument('--name');p.add_argument('--engine',choices=['original','wide'],default='wide');p.add_argument('--input',choices=['legacy_960','native_960','native_1440','native_1440_color'],default='native_1440');p.add_argument('--steps',type=int,default=4000);p.add_argument('--interval',type=int,default=500);p.add_argument('--lr',type=float,default=1.);p.add_argument('--densify',action='store_true');a=p.parse_args();prepare() if a.stage=='prepare' else train(a)
