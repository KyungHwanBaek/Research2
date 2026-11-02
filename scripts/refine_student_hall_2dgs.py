#!/usr/bin/env python3
"""Matched Metal 2DGS runs, including genuine differentiable LiDAR depth loss.

Common seed, views, initialization, normals and budget. Validation images are
excluded from RGB/depth losses and seed creation, but used in original poses.
This is a project-specific 2DGS experiment, not a paper reproduction.
"""
from pathlib import Path
import argparse,json,time,os
import numpy as np
import cv2
from scipy.spatial.transform import Rotation

BASE=Path('experiments/student_hall_lidar')
OUT=Path('experiments/student_hall_refinement_20261026')
VAL=[20,26,32,38,44]
INDICES=list(range(20,46))
TRAIN=[i for i in INDICES if i not in VAL]

def prepare():
    poses=np.load(BASE/'joint_poses.npz')['poses'];K=np.load(BASE/'calibration.npz')['K'];rng=np.random.default_rng(42)
    xs=[];cs=[];rots=[];scales=[]
    # Small smooth LiDAR triangles are the seed only; exact raw sparse depths
    # are used for the training depth loss (not interpolated targets).
    for i in TRAIN:
        d=np.load(BASE/'frames'/f'{i:06d}.npz')['depth'];im=cv2.imread(str(BASE/'images'/f'{i:06d}.png'))[:,:,::-1]
        y,x=np.mgrid[3:717:3,3:957:3];z=d[y,x]
        ok=(z>3)&(z<32)
        xyz=np.stack(((x-K[0,2])/K[0,0]*z,(y-K[1,2])/K[1,1]*z,z),axis=-1)
        dx=xyz[1:-1,2:]-xyz[1:-1,:-2];dy=xyz[2:,1:-1]-xyz[:-2,1:-1]
        n=np.cross(dx,dy);n/=np.maximum(np.linalg.norm(n,axis=-1,keepdims=True),1e-9)
        valid=ok[1:-1,1:-1]&ok[1:-1,2:]&ok[1:-1,:-2]&ok[2:,1:-1]&ok[:-2,1:-1]
        valid&=np.abs(d[y[1:-1,2:],x[1:-1,2:]]-d[y[1:-1,:-2],x[1:-1,:-2]])<(.1+.03*z[1:-1,1:-1])
        p=xyz[1:-1,1:-1][valid];n=n[valid];zz=p[:,2]
        # Construct a proper right-handed tangent frame for wxyz quaternions.
        u=np.cross(np.tile([0,1,0],(len(n),1)),n);un=np.linalg.norm(u,axis=1);bad=un<.05
        u[bad]=np.cross(np.tile([1,0,0],(bad.sum(),1)),n[bad]);u/=np.maximum(np.linalg.norm(u,axis=1,keepdims=True),1e-9)
        v=np.cross(n,u);rot=np.stack((u,v,n),axis=-1);rot=poses[i,:3,:3][None]@rot
        q=Rotation.from_matrix(rot).as_quat();q=np.column_stack((q[:,3],q[:,:3]))
        xs.append(p@poses[i,:3,:3].T+poses[i,:3,3]);cs.append(im[y[1:-1,1:-1][valid],x[1:-1,1:-1][valid]])
        rots.append(q);scales.append(np.maximum(.012,zz/K[0,0]*2.5))
    x=np.concatenate(xs);c=np.concatenate(cs);q=np.concatenate(rots);s=np.concatenate(scales)
    # Retain real per-view surfels rather than averaged points across walls.
    vox=np.floor(x/.045).astype(np.int32);_,keep=np.unique(vox,axis=0,return_index=True)
    if len(keep)>100000:keep=rng.choice(keep,100000,replace=False)
    np.savez_compressed(OUT/'surfel_initialization.npz',points=x[keep].astype(np.float32),rgb=c[keep],quats=q[keep].astype(np.float32),scales=s[keep].astype(np.float32),poses=poses,K=K)
    print('surfel_seed',len(keep),'train',TRAIN,'validation',VAL,flush=True)

def main(a):
    if a.prepare:return prepare()
    import torch
    from metalsplat import Gaussian2DModel,Camera,render_2dgs,save_ply_2dgs
    from metalsplat.losses import gaussian_splatting_loss,normal_consistency_loss,distortion_loss
    from metalsplat.training import GaussianTrainingState,save_image
    from metalsplat.optim import SparseAdam,sh_lr_scale
    from metalsplat.densify2dgs import densify_and_prune_2dgs
    from skimage.metrics import structural_similarity
    torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42)
    assert torch.backends.mps.is_available()
    device='mps';z=np.load(OUT/'surfel_initialization.npz');poses=z['poses'];K=z['K'].copy();scale=a.width/960;K[:2]*=scale
    height=int(a.width*.75);folder=OUT/f'gs2d_{"warm_" if a.warmstart else ""}{a.method}_{a.width}_{a.steps}';folder.mkdir(exist_ok=True)
    render_scale=1.;center=np.zeros(3)
    if a.warmstart:
        # Read text outside pycolmap: its bundled OpenMP conflicts with torch's
        # runtime in this environment. Do not enable duplicate-runtime bypass.
        lines=(BASE/'gs_colmap/sparse_text/images.txt').read_text().splitlines()
        for line in lines[::2]:
            a0=line.split();q=np.array(a0[1:5],float);T=np.eye(4);T[:3,:3]=Rotation.from_quat(np.r_[q[1:],q[0]]).as_matrix();T[:3,3]=np.array(a0[5:8],float)
            poses[int(Path(a0[9]).stem)]=np.linalg.inv(T)
        centers=poses[INDICES,:3,3];center=centers.mean(0);render_scale=1/np.max(np.abs(centers-center))
        poses=poses.copy();poses[:,:3,3]=(poses[:,:3,3]-center)*render_scale
    def tensor(x):return torch.from_numpy(np.ascontiguousarray(x)).float().to(device)
    cameras={};images={};depths={}
    for i in INDICES:
        T=np.linalg.inv(poses[i]);cameras[i]=Camera(tensor(T[:3,:3]),tensor(T[:3,3]),K[0,0],K[1,1],K[0,2],K[1,2],a.width,height)
        im=cv2.imread(str(BASE/'images'/f'{i:06d}.png'))[:,:,::-1];im=cv2.resize(im,(a.width,height),interpolation=cv2.INTER_AREA)
        images[i]=tensor(im/255)
        # Rasterize exact measured raw LiDAR into training resolution; choose
        # nearest depth at each pixel and remove rig self-returns <3m.
        x=np.load(BASE/'frames'/f'{i:06d}.npz')['camera'];x=x[(np.linalg.norm(x,axis=1)>3)&(x[:,2]>1)&(x[:,2]<35)]
        uv=np.rint(x[:,:2]/x[:,2:]*[K[0,0],K[1,1]]+K[:2,2]).astype(int)
        ok=(uv[:,0]>=0)&(uv[:,0]<a.width)&(uv[:,1]>=0)&(uv[:,1]<height);uv=uv[ok];x=x[ok]
        depth=np.full(height*a.width,np.inf,np.float32);np.minimum.at(depth,uv[:,1]*a.width+uv[:,0],x[:,2]);depth[~np.isfinite(depth)]=0
        depths[i]=tensor(depth.reshape(height,a.width))
    if a.warmstart:
        from metalsplat.export import load_ply
        original=load_ply(BASE/'gs_colmap_12000/scene.ply')
        from metalsplat.utils.quaternion import quat_to_rotmat
        mats=quat_to_rotmat(original.quats).detach().numpy();sc=original.scales.detach().numpy();order=np.argsort(-sc,axis=1)
        mats=np.take_along_axis(mats,order[:,None,:],axis=2);mats[:,:,2]=np.cross(mats[:,:,0],mats[:,:,1])
        quat=Rotation.from_matrix(mats).as_quat();quat=np.column_stack((quat[:,3],quat[:,:3]))
        sizes=np.take_along_axis(sc,order,axis=1)[:,:2]
        model=Gaussian2DModel(original.means.detach(),scales=torch.from_numpy(sizes.copy()),quats=torch.from_numpy(quat.copy()).float(),
            opacities=original.opacities.detach(),sh_degree=3,sh_coeffs=original.raw_sh.detach()).to(device)
        del original
    else:
        model=Gaussian2DModel(tensor(z['points']),scales=tensor(np.repeat(z['scales'][:,None],2,axis=1)),
            quats=tensor(z['quats']),colors=tensor(z['rgb']/255),opacities=torch.full((len(z['points']),),.2,device=device),sh_degree=3).to(device)
        model.active_sh_degree=0
    lr=.00001 if a.warmstart else .0015
    def optimizer(m):return SparseAdam([
        {'params':[m.means],'lr':lr}, {'params':[m.raw_sh],'lr':.0025,'lr_scale':sh_lr_scale(m.raw_sh)},
        {'params':[m.raw_scales],'lr':.003},{'params':[m.raw_quats],'lr':.001},{'params':[m.raw_opacities],'lr':.03}])
    state=GaussianTrainingState(model,optimizer,track_pixel_count=True);threshold=None;best=-np.inf;history=[];start=time.time()
    manifest=dict(method=a.method,steps=a.steps,width=a.width,train=TRAIN,validation=VAL,seed=42,initial_surfels=model.num_points,
        poses='previous joint photometric + LiDAR refinement, all source views used',depth_loss_weight=.15 if a.method=='depth' else 0,
        depth_loss='absolute log expected depth ratio capped 0.3; alpha-normalized differentiable sum, transmittance detached by renderer',
        caveat='Validation views excluded from this training and seed; pose/SfM estimation already used all images. No independent survey ground truth.')
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2))
    def evaluate(step):
        nonlocal best
        rows=[]
        with torch.no_grad():
            for i in VAL:
                aux=render_2dgs(state.model,cameras[i],return_aux=True);pred=aux.image.clamp(0,1).cpu().numpy();truth=images[i].cpu().numpy()
                mse=np.mean((pred-truth)**2);psnr=float(-10*np.log10(mse));ssim=float(structural_similarity(truth,pred,data_range=1,channel_axis=2))
                alpha=1-aux.final_T;dep=(aux.depth/alpha.clamp_min(1e-6)/render_scale).cpu().numpy();ref=depths[i].cpu().numpy()
                ok=(ref>0)&(alpha.cpu().numpy()>.5);e=np.abs(dep[ok]-ref[ok]);rel=e/ref[ok]
                rows.append(dict(frame=i,psnr=psnr,ssim=ssim,depth_samples=int(ok.sum()),depth_median_cm=float(np.median(e)*100),depth_median_relative=float(np.median(rel))))
                save_image(aux.image,folder/f'validation_{i:02d}_latest.png')
        report=dict(step=step,gaussians=state.model.num_points,elapsed_s=time.time()-start,psnr=float(np.mean([r['psnr'] for r in rows])),ssim=float(np.mean([r['ssim'] for r in rows])),views=rows)
        history.append(report);(folder/'history.json').write_text(json.dumps(history,indent=2));print('evaluation',json.dumps(report),flush=True)
        if report['psnr']>best:
            best=report['psnr'];save_ply_2dgs(state.model,folder/'best.ply');(folder/'best_metrics.json').write_text(json.dumps(report,indent=2))
            torch.save(dict(raw={k:v.cpu() for k,v in state.model.raw_rows().items()},active_sh_degree=state.model.active_sh_degree,step=step),folder/'best.pt')
            for i in VAL:
                import shutil
                shutil.copyfile(folder/f'validation_{i:02d}_latest.png',folder/f'validation_{i:02d}_best.png')
    evaluate(0)
    order=[]
    for step in range(1,a.steps+1):
        if not order:order=np.random.permutation(TRAIN).tolist()
        i=order.pop();state.optimizer.param_groups[0]['lr']=lr*(.02**(step/a.steps))
        state.optimizer.zero_grad();aux=render_2dgs(state.model,cameras[i],return_aux=True,abs_grad_accum=state.grad_accum,pixel_count_accum=state.pixel_count)
        photo=gaussian_splatting_loss(aux.image,images[i],lambda_dssim=.2);loss=photo;dl=0.
        if a.method=='depth':
            alpha=1-aux.final_T;mask=(depths[i]>0)&(alpha>.3)
            pred=aux.depth/alpha.clamp_min(1e-6)/render_scale
            dl=torch.abs(torch.log(pred[mask].clamp_min(.1)/depths[i][mask])).clamp_max(.3).mean();loss=loss+.15*dl
        if step>800:loss=loss+100*distortion_loss(aux.distortion)
        if step>1200:loss=loss+.01*normal_consistency_loss(aux.normal,aux.depth,1-aux.final_T,cameras[i])
        if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite loss at {step}')
        loss.backward();visible=aux.valid>.5;state.optimizer.step(visible);state.record_visibility(visible)
        if step%100==0 or step==1:
            torch.mps.synchronize();print('step',step,'rgb',photo.item(),'depth',float(dl) if isinstance(dl,float) else dl.item(),'n',state.model.num_points,'seconds',round(time.time()-start,1),flush=True)
        if 300<=step<=int(a.steps*.65) and step%100==0 and state.model.num_points<300000:
            new,stats=densify_and_prune_2dgs(state.model,state.grad_accum,state.grad_count,pixel_count=state.pixel_count,
                grad_percentile=.85,prune_opacity_thresh=.01,grad_threshold=threshold,max_world_size=.7)
            if threshold is None and stats.grad_threshold is not None:threshold=stats.grad_threshold
            state.replace_model(new,stats.source_index,stats.parent_index);state.reset_accumulators()
            print('densify',step,stats.n_before,stats.n_after,flush=True)
        if step in [500,1000,1500] and not a.warmstart:state.model.increase_sh_degree()
        if step%a.eval_every==0 or step==a.steps:evaluate(step)
    save_ply_2dgs(state.model,folder/'final.ply')
    print('completed',str(folder),'best_psnr',best,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--warmstart',action='store_true');p.add_argument('--method',choices=['rgb','depth'],default='depth');p.add_argument('--width',type=int,default=720);p.add_argument('--steps',type=int,default=4000);p.add_argument('--eval-every',type=int,default=1000);main(p.parse_args())
