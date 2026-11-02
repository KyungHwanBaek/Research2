#!/usr/bin/env python3
"""Scientific consistency, train/validation split, assets and preservation QA."""
from pathlib import Path
from html.parser import HTMLParser
import json,subprocess
import numpy as np
from PIL import Image
import pycolmap
from hall_research_experiments import O,P,VAL,read,dump
from hall_research_publish import NAMES,SITE,sha,score
def main():
    result=read(O/'results.json');checks={};fields=[]
    for name in ['baseline_render']+NAMES:
        m=read(O/name/'best_metrics.json');s=score(O/name);assert abs(s['psnr']-m['psnr'])<1e-10;assert abs(s['ssim']-m['ssim'])<1e-10
        assert np.isfinite([s['psnr'],s['ssim']]).all()
        for k in range(5):assert Image.open(O/name/f'heldout_{k:02d}.png').size==(720,540)
        # Parse the actual binary PLY; all stored Gaussian coefficients must be finite.
        path=O/name/'best.ply'
        with path.open('rb') as f:
            lines=[]
            while True:
                line=f.readline().decode('ascii');lines.append(line)
                if line.strip()=='end_header':break
            properties=[line for line in lines if line.startswith('property ')];assert all(line.startswith('property float ') for line in properties)
            count=int(next(x for x in lines if x.startswith('element vertex ')).split()[-1]);arr=np.fromfile(f,dtype='<f4');assert arr.size==count*len(properties);assert np.isfinite(arr).all();assert count==m['gaussians']
        fields.append(dict(name=name,gaussians=count,properties=len(properties),finite=True))
    checks['gaussian_models']=fields
    for name in ['native_960','native_1440','native_1440_color']:
        meta=read(O/name/'manifest.json');train=[x for x in meta['images'] if x['split']=='train'];val=[x for x in meta['images'] if x['split']=='validation'];assert len(train)==42 and len(val)==5;assert not set(VAL)&{x['frame'] for x in train};assert [x['frame'] for x in val]==VAL
        r=pycolmap.Reconstruction(O/name/'sparse/0');assert r.num_reg_images()==47
        for im in r.images.values():
            c=r.cameras[im.camera_id];assert Image.open(O/name/'images'/im.name).size==(c.width,c.height)
    fits=read(O/'stereo_color_fits.json')['fits'];assert len(fits)==21;assert not set(VAL)&{int(Path(x['name']).stem) for x in fits}
    checks['training_split_and_color_fit']=True
    for name in ['native960_low_lr','native1440_low_lr','native1440_stereo_color','native1440_progressive_split']:
        h=read(O/name/'history.json');assert [v['step'] for v in h]==list(range(5000,9001,500));assert all(v['validation_overflow_flags']==[0]*5 for v in h)
    checks['actual_additional_training_iterations']=16000;checks['all_recorded_validation_overflow_flags_zero']=True
    expected_good=[r for r in result['candidates'] if r['psnr']>=result['baseline']['psnr']+.05 and r['ssim']>=result['baseline']['ssim']-.002];assert bool(expected_good)==result['adopted'];checks['selection_rule_obeyed']=True
    for name in ['baseline_render','native1440_low_lr']:
        m=read(O/name/'right_probe_metrics.json');assert not m['checkpoint_selection_used_these_images'];assert m['pose_reproduction_max_abs_error']<1e-5
    checks['right_probe_without_pose_or_exposure_fit']=True
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.refs=[]
        def handle_starttag(self,tag,attrs):
            for k,v in attrs:
                if k in ['href','src','poster'] and v:self.refs.append(v)
    page=SITE/'student-hall-research.html';p=Links();p.feed(page.read_text());missing=[]
    for ref in p.refs:
        if ref.startswith(('#','https://','http://','data:')):continue
        if ref.endswith('_records.zip'):continue # assembled after QA; independently checked during packaging.
        if not (SITE/ref.split('#')[0]).exists():missing.append(ref)
    assert not missing,missing;checks['local_page_asset_links']=True
    for file in SITE.rglob('*'):
        if file.is_file():assert file.stat().st_size<24*1024*1024,(str(file),file.stat().st_size)
    checks['individual_public_files_under_24MiB']=True
    v=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','stream=width,height,nb_frames','-of','json',str(SITE/'assets/hall-research-flythrough.mp4')],text=True));assert v['streams'][0]==dict(width=960,height=720,nb_frames='72');checks['video']=v['streams'][0]
    for row in read(O/'baseline_manifest.json'):assert sha(Path(row['path']))==row['sha256']
    checks['previous_selected_model_preserved']=True
    checks['browser_ui_test_performed']=False;checks['browser_limitation']='Runtime selection: no browser available; discovery list empty. No alternative automation used.'
    checks['all_passed']=True;dump(O/'qa.json',checks);print('QA_PASSED',json.dumps(checks),flush=True)
if __name__=='__main__':main()
