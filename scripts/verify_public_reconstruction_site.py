#!/usr/bin/env python3
"""Check public asset completeness, geometry integrity and measured improvements."""
from pathlib import Path
from html.parser import HTMLParser
import json,subprocess
import numpy as np


def main():
    site=Path('sites/reconstruction-lab-share/dist');assets=site/'assets';data=json.loads((assets/'scene.json').read_text())
    errors=[]
    class Links(HTMLParser):
        def handle_starttag(self,tag,attrs):
            for key,value in attrs:
                if key in ['href','src','poster'] and value and not value.startswith(('http:','https:','#','data:')):
                    if not (site/value).is_file():errors.append(value)
    parser=Links();parser.feed((site/'index.html').read_text());assert not errors,errors
    for name in ['app.js','viewer.js']:subprocess.run(['node','--check',str(site/name)],check=True)
    reports=[]
    for s in data['segments']:
        i=s['id'];pos=np.fromfile(assets/f's{i}-points.bin','<f4');col=np.fromfile(assets/f's{i}-colors.bin','u1')
        assert len(pos)==len(col)==s['displayPoints']*3 and np.isfinite(pos).all()
        mesh=np.fromfile(assets/f's{i}-mesh-vertices.bin','<f4');norm=np.fromfile(assets/f's{i}-mesh-normals.bin','<f4')
        colors=np.fromfile(assets/f's{i}-mesh-colors.bin','u1');faces=np.fromfile(assets/f's{i}-mesh-faces.bin','<u4')
        assert len(mesh)==len(norm)==len(colors) and len(mesh)%3==0 and np.isfinite(mesh).all() and np.isfinite(norm).all()
        assert faces.max()<len(mesh)//3 and len(faces)==s['displayTriangles']*3
        for photo in s['photos']:assert (site/photo['src']).is_file()
        assert (site/s['download']).is_file()
        reports.append({'segment':i+1,'display_points':s['displayPoints'],'display_triangles':s['displayTriangles'],'source_photos':len(s['photos'])})
    for ex in data['evaluationExamples']:
        for kind in ['input','render']:assert (assets/f'eval-{ex["index"]}-{kind}.jpg').is_file()
    files=[p for p in site.rglob('*') if p.is_file()]
    largest=max(files,key=lambda p:p.stat().st_size);assert largest.stat().st_size<24*1024**2
    probe=subprocess.run(['ffprobe','-v','error','-show_entries','stream=codec_name,width,height','-of','json',str(assets/'flythrough.mp4')],check=True,capture_output=True,text=True)
    assert json.loads(probe.stdout)['streams'][0]['codec_name']=='h264'
    poses=np.load('experiments/stereo_joint_ba_0905/selected_poses.npz')['poses']
    assert np.isfinite(poses).all() and np.allclose(np.linalg.det(poses[:,:3,:3]),1)
    ba=json.load(open('experiments/stereo_joint_ba_0905/comparison.json'))
    for r in ba:
        assert r['summary']['termination_type']=='BundleAdjustmentTerminationType.CONVERGENCE'
        assert r['selected_dense_score']['median_cross_view_relative_depth_error']<=r['dense_before']['median_cross_view_relative_depth_error']
        assert r['selected_dense_score']['mean_cross_view_agreement_fraction']>=r['dense_before']['mean_cross_view_agreement_fraction']
    result={'passed':True,'segments':reports,'site_files':len(files),'total_asset_bytes':sum(p.stat().st_size for p in files),
        'largest_asset_bytes':largest.stat().st_size,'native_rig_ba':'all three segments converged',
        'independent_dense_pixels':'selected pose blends improve both recorded consistency diagnostics',
        'viewer_syntax_and_buffers':'passed','video':json.loads(probe.stdout),
        'browser_visual_QA':'unavailable: Browser runtime listed no available browsers',
        'absolute_geometry_accuracy':'not independently established'}
    Path('experiments/stereo_joint_ba_0905/public_site_verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
