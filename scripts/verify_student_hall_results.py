#!/usr/bin/env python3
"""Artifact checks for actual reconstructions and public static assets."""
from pathlib import Path
from html.parser import HTMLParser
import json,re,subprocess
import numpy as np

ROOT=Path('experiments/student_hall_lidar');SITE=Path('sites/reconstruction-lab-share/dist');ASSETS=SITE/'assets'

def main():
    data=json.loads((ASSETS/'student-hall.json').read_text());stats=json.loads((ROOT/'fused/summary.json').read_text())
    assert stats==data['summary']
    assert stats['frames']==48 and stats['colored_points']>200000 and stats['full_map_points']>1000000
    errors=[]
    class Links(HTMLParser):
        def handle_starttag(self,tag,attrs):
            for key,value in attrs:
                if key in ['src','href','poster'] and value and not value.startswith(('http:','https:','#','data:')):
                    if not (SITE/value.split('?')[0]).is_file():errors.append(value)
    parser=Links();html=(SITE/'student-hall.html').read_text();parser.feed(html);assert not errors,errors
    ids=set(re.findall(r'id="([^"]+)"',html));js=(SITE/'student-hall.js').read_text()
    assert set(re.findall(r"byId\('([^']+)'\)",js))<=ids
    for name in ['student-hall.js','student-viewer.js']:subprocess.run(['node','--check',str(SITE/name)],check=True)
    buffers=[]
    for area in data['areas']:
        seg=area['segment']
        for i in [seg,seg+2,seg+4]:
            x=np.fromfile(ASSETS/f's{i}-points.bin','<f4');c=np.fromfile(ASSETS/f's{i}-colors.bin','u1')
            assert x.size%3==0 and x.size==c.size and np.isfinite(x).all()
            assert (ASSETS/f's{i}-sample.ply').is_file();buffers.append({'id':i,'points':x.size//3})
        x=np.fromfile(ASSETS/f's{seg}-mesh-vertices.bin','<f4');n=np.fromfile(ASSETS/f's{seg}-mesh-normals.bin','<f4')
        c=np.fromfile(ASSETS/f's{seg}-mesh-colors.bin','u1');f=np.fromfile(ASSETS/f's{seg}-mesh-faces.bin','<u4')
        assert x.size==n.size==c.size and np.isfinite(x).all() and np.isfinite(n).all()
        assert f.max()<x.size//3 and f.size==area['triangles']*3
        assert (ASSETS/f's{seg}-mesh.ply').is_file()
    for name in ['kiss','posegraph','joint','sensor_clean']:
        p=np.load(ROOT/f'{name}_poses.npz')['poses'];assert p.shape==(48,4,4) and np.isfinite(p).all()
        assert np.allclose(np.linalg.det(p[:,:3,:3]),1.,atol=1e-5)
    s=np.load(ROOT/'fused/selected_poses.npz')['poses'];assert np.isfinite(s).all()
    joint=json.loads((ROOT/'joint_report.json').read_text());assert joint['success'] and joint['after_median_px']<joint['before_median_px']
    for photo in data['photos']:assert (SITE/photo['src']).is_file()
    for k in range(5):
        for kind in ['input','render']:assert (ASSETS/f'hall-eval-{k}-{kind}.jpg').is_file()
    trials=data['renderingTrials'];best=max(trials,key=lambda x:x['metrics']['psnr']);assert best['metrics']==data['rendering']['metrics']
    probe=subprocess.run(['ffprobe','-v','error','-show_entries','stream=codec_name,width,height','-of','json',str(ASSETS/'hall-flythrough.mp4')],capture_output=True,text=True,check=True)
    video=json.loads(probe.stdout)['streams'][0];assert video['codec_name']=='h264' and video['width']==720
    files=[x for x in SITE.rglob('*') if x.is_file()];largest=max(files,key=lambda x:x.stat().st_size)
    assert largest.stat().st_size<24*1024**2,(str(largest),largest.stat().st_size)
    assert (SITE/'student-hall-report.md').read_text()==Path('STUDENT_HALL_LIDAR_REPORT.md').read_text()
    assert (ROOT/'geometry_contact_sheet.jpg').is_file()
    report={'passed':True,'summary':stats,'display_buffers':buffers,'rendering':data['rendering'],
        'largest_site_asset':str(largest),'largest_asset_bytes':largest.stat().st_size,'video':video,
        'site_files':len(files),'site_bytes':sum(p.stat().st_size for p in files),
        'geometry_render_QA':'Three reconstructed camera views inspected; near-sensor false occluder removed.',
        'browser_interaction_QA':'Unavailable: browser runtime returned no available browser.',
        'absolute_accuracy':'Not established by independent ground truth.'}
    (ROOT/'verification.json').write_text(json.dumps(report,indent=2,ensure_ascii=False));print(json.dumps(report,indent=2,ensure_ascii=False))

if __name__=='__main__':main()
